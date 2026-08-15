"""Write-once RU calibration-pair materialization and audio-isolation gate for v4.

This module consumes the metadata-only calibration selection, but deliberately does
not load a detector checkpoint, fit a temperature, score a model, or produce a final
prediction.  It binds the selected VoxForge WAVs to the pinned archive, creates one
new text-only eSpeak derivative per surviving source item, and makes a complete pair
lock only after technical QA plus exact/near-audio leakage screening.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import wave
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, resolve_asset_path, sha256_file
from kds.data.espeakng import (
    EspeakNgProfile,
    extract_verified_espeakng_runtime,
    load_espeakng_runtime,
    synthesize_espeakng,
)
from kds.data.licenses import (
    APPROVED_LICENSE_STATUSES,
    load_license_ledger,
    validate_manifest_licenses,
)
from kds.data.manifest import (
    REQUIRED_FIELDS,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.research_tts import (
    ResearchTtsModel,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.data.v4_audio_gate import (
    V4AudioSignature,
    V4DecodedCandidate,
    V4DecodedDecision,
    V4DecodeResult,
    V4DecodeTask,
    append_v4_decode_journal,
    decide_v4_decoded_audio_eligibility,
    load_v4_decode_journal,
    run_v4_decode_task,
)
from kds.data.voxforge import (
    VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
    VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
    VOXFORGE_RU_ARCHIVE_ROOT,
    VOXFORGE_RU_LICENSE,
    VoxForgeRuRecord,
    load_voxforge_ru_metadata,
)
from kds.eval.voxforge_metadata_screen import voxforge_metadata_identity

PROTOCOL_ID = "xlsr-sls-model-v4-calibration-materialization-v1"
PLAN_SCHEMA_VERSION = 1
SOURCE_ID = "voxforge_ru_mdc_2026_05_v4_calibration"
SPOOF_ID = "voxforge_ru_mdc_2026_05_v4_calibration_espeakng"
SELECTION_FIELDS = (
    "selection_rank",
    "sample_id",
    "submission_pseudo_id",
    "prompt_id",
    "parent_group_id",
    "speaker_pseudo_id",
    "prompt_text_hash",
    "original_prompt_text_hash",
)
INVENTORY_FIELDS = (
    "kind",
    "selection_rank",
    "sample_id",
    "paired_source_sample_id",
    "text_id",
    "text_hash",
    "raw_relative_path",
    "raw_audio_sha256",
    "decoded_relative_path",
    "decoded_audio_sha256",
    "preparation_status",
    "eligibility_status",
    "rejection_reason",
    "historical_raw_exact_match_count",
    "historical_decoded_exact_match_count",
    "historical_near_match_count",
    "within_pool_near_match_count",
)
_HEX = frozenset("0123456789abcdef")


class V4CalibrationMaterializationError(ValueError):
    """Raised when this one-shot materialization cannot prove every required boundary."""


@dataclass(frozen=True, slots=True)
class Binding:
    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, Binding]
    archive_name: str
    archive_size_bytes: int
    archive_sha256: str
    model_id: str
    model_root: str
    target_rows: int
    raw_source_root: str
    raw_spoof_root: str
    processed_source_root: str
    processed_spoof_root: str
    runtime_root: str
    outputs: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SelectionRow:
    selection_rank: int
    sample_id: str
    submission_pseudo_id: str
    prompt_id: str
    parent_group_id: str
    speaker_pseudo_id: str
    prompt_text_hash: str
    original_prompt_text_hash: str


@dataclass(frozen=True, slots=True)
class BoundSelection:
    selection: SelectionRow
    record: VoxForgeRuRecord


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4CalibrationMaterializationError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4CalibrationMaterializationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4CalibrationMaterializationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4CalibrationMaterializationError(
            f"{label} must be a non-empty project-relative path."
        )
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise V4CalibrationMaterializationError(f"{label} is not a safe project-relative path.")
    return parsed.as_posix()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise V4CalibrationMaterializationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4CalibrationMaterializationError(f"{label} must be a positive integer.")
    return value


def _binding(value: object, label: str) -> Binding:
    raw = _mapping(value, label)
    if set(raw) != {"path", "sha256", "rows"}:
        raise V4CalibrationMaterializationError(f"{label} must contain path, sha256 and rows.")
    rows = raw["rows"]
    if rows is not None:
        _positive_int(rows, f"{label}.rows")
    return Binding(
        path=_safe_path(raw["path"], f"{label}.path"),
        sha256=_sha256(raw["sha256"], f"{label}.sha256"),
        rows=cast(int | None, rows),
    )


def _project_path(project_root: Path, relative: str, label: str) -> Path:
    candidate = (project_root / relative).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise V4CalibrationMaterializationError(
            f"{label} resolves outside the project root."
        ) from error
    return candidate


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4CalibrationMaterializationError(f"Cannot count CSV rows in {path}.") from error


def _binding_rows(path: Path) -> int:
    if path.suffix == ".jsonl":
        try:
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
        except (OSError, UnicodeDecodeError) as error:
            raise V4CalibrationMaterializationError(
                f"Cannot count JSONL rows in {path}."
            ) from error
    return _csv_rows(path)


def _verify_binding(binding: Binding, project_root: Path, label: str) -> Path:
    path = _project_path(project_root, binding.path, label)
    if not path.is_file() or sha256_file(path) != binding.sha256:
        raise V4CalibrationMaterializationError(f"{label} binding does not match: {binding.path}")
    if binding.rows is not None and _binding_rows(path) != binding.rows:
        raise V4CalibrationMaterializationError(f"{label} row count changed: {binding.path}")
    return path


def load_calibration_materialization_plan(path: Path, project_root: Path) -> CalibrationPlan:
    """Load the pre-execution contract and verify all static hash-pinned inputs."""

    root = project_root.resolve(strict=True)
    plan_path = path.resolve(strict=True)
    try:
        relative_plan = plan_path.relative_to(root).as_posix()
    except ValueError as error:
        raise V4CalibrationMaterializationError(
            "Materialization plan must be inside project root."
        ) from error
    raw = _json_object(plan_path, "v4 calibration materialization plan")
    expected_top = {
        "schema_version",
        "protocol_id",
        "created_at",
        "inputs",
        "source",
        "working",
        "outputs",
        "prohibitions",
    }
    if (
        set(raw) != expected_top
        or raw["schema_version"] != PLAN_SCHEMA_VERSION
        or raw["protocol_id"] != PROTOCOL_ID
    ):
        raise V4CalibrationMaterializationError("Materialization plan schema/protocol is invalid.")
    created_at = raw["created_at"]
    if not isinstance(created_at, str):
        raise V4CalibrationMaterializationError("Materialization plan timestamp is invalid.")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4CalibrationMaterializationError(
            "Materialization plan timestamp is invalid."
        ) from error
    raw_inputs = _mapping(raw["inputs"], "inputs")
    required_inputs = {
        "calibration_input_plan",
        "calibration_input_receipt",
        "metadata_selection",
        "frozen_license_ledger",
        "voxforge_source_audit",
        "espeak_model_lock",
        "v4_train_manifest",
        "v4_dev_manifest",
        "historical_fingerprint_inventory",
        "source_decode_inventory",
        "kk_spoof_decode_inventory",
        "dev_source_decode_journal",
        "dev_spoof_decode_journal",
        "calibration_module",
        "voxforge_module",
        "espeak_module",
        "audio_gate_module",
        "materialization_module",
        "runner_script",
    }
    if set(raw_inputs) != required_inputs:
        raise V4CalibrationMaterializationError("Materialization plan inputs are incomplete.")
    inputs = {name: _binding(value, f"inputs.{name}") for name, value in raw_inputs.items()}
    source = _mapping(raw["source"], "source")
    if set(source) != {
        "archive_name",
        "archive_size_bytes",
        "archive_sha256",
        "espeak_model_id",
        "target_rows",
    }:
        raise V4CalibrationMaterializationError("Materialization source binding is invalid.")
    archive_name = source["archive_name"]
    if not isinstance(archive_name, str) or not archive_name:
        raise V4CalibrationMaterializationError("source.archive_name must be non-empty.")
    archive_size = _positive_int(source["archive_size_bytes"], "source.archive_size_bytes")
    archive_sha = _sha256(source["archive_sha256"], "source.archive_sha256")
    model_id = source["espeak_model_id"]
    if not isinstance(model_id, str) or not model_id:
        raise V4CalibrationMaterializationError("source.espeak_model_id must be non-empty.")
    target_rows = _positive_int(source["target_rows"], "source.target_rows")
    if (
        archive_size != VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive_sha != VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256
    ):
        raise V4CalibrationMaterializationError(
            "Materialization must retain the pinned VoxForge archive."
        )
    working = _mapping(raw["working"], "working")
    if set(working) != {
        "raw_source_root",
        "raw_spoof_root",
        "processed_source_root",
        "processed_spoof_root",
        "runtime_root",
        "model_root",
    }:
        raise V4CalibrationMaterializationError("Materialization working paths are invalid.")
    values = {name: _safe_path(value, f"working.{name}") for name, value in working.items()}
    if len(set(values.values())) != len(values):
        raise V4CalibrationMaterializationError("Materialization working paths must be distinct.")
    raw_outputs = _mapping(raw["outputs"], "outputs")
    required_outputs = {
        "source_raw_manifest",
        "source_ready_manifest",
        "spoof_raw_manifest",
        "spoof_ready_manifest",
        "audio_gate_inventory",
        "pair_lock_manifest",
        "receipt",
    }
    if set(raw_outputs) != required_outputs:
        raise V4CalibrationMaterializationError("Materialization output set is invalid.")
    outputs = {name: _safe_path(value, f"outputs.{name}") for name, value in raw_outputs.items()}
    if len(set(outputs.values())) != len(outputs):
        raise V4CalibrationMaterializationError("Materialization output paths must be distinct.")
    prohibitions = _mapping(raw["prohibitions"], "prohibitions")
    expected_prohibitions = {
        "network_downloads": True,
        "checkpoint_loading": True,
        "calibration": True,
        "temperature_fitting": True,
        "final_inference": True,
        "detector_inference": True,
        "detector_feedback": True,
        "acoustic_review": True,
        "output_overwrite": True,
        "resynthesis": True,
        "replacement_or_backfill": True,
    }
    if prohibitions != expected_prohibitions:
        raise V4CalibrationMaterializationError("Materialization prohibitions are not fail-closed.")
    for name, binding in inputs.items():
        _verify_binding(binding, root, f"inputs.{name}")
    return CalibrationPlan(
        path=relative_plan,
        sha256=sha256_file(plan_path),
        created_at=created_at,
        inputs=inputs,
        archive_name=archive_name,
        archive_size_bytes=archive_size,
        archive_sha256=archive_sha,
        model_id=model_id,
        model_root=values["model_root"],
        target_rows=target_rows,
        raw_source_root=values["raw_source_root"],
        raw_spoof_root=values["raw_spoof_root"],
        processed_source_root=values["processed_source_root"],
        processed_spoof_root=values["processed_spoof_root"],
        runtime_root=values["runtime_root"],
        outputs=outputs,
    )


def _load_selection(plan: CalibrationPlan, project_root: Path) -> tuple[SelectionRow, ...]:
    """Bind the immutable metadata-only receipt and parse its 81 privacy-safe identities."""

    receipt_path = _verify_binding(
        plan.inputs["calibration_input_receipt"], project_root, "metadata receipt"
    )
    receipt = _json_object(receipt_path, "calibration metadata receipt")
    outputs = _mapping(receipt.get("outputs"), "metadata receipt outputs")
    selected = _mapping(outputs.get("metadata_selection"), "metadata receipt selected output")
    claims = _mapping(receipt.get("claims"), "metadata receipt claims")
    if (
        receipt.get("protocol_id") != "xlsr-sls-model-v4-calibration-inputs-v1"
        or receipt.get("status") != "metadata_inputs_frozen_materialization_contract_required"
        or selected.get("path") != plan.inputs["metadata_selection"].path
        or selected.get("sha256") != plan.inputs["metadata_selection"].sha256
        or selected.get("rows") != plan.target_rows
        or claims.get("v4_train_dev_isolation_passed") is not True
        or claims.get("raw_audio_extraction_performed") is not False
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("calibration_performed") is not False
        or claims.get("checkpoint_loaded") is not False
    ):
        raise V4CalibrationMaterializationError(
            "Metadata gate does not authorize this materialization boundary."
        )
    selection_path = _verify_binding(
        plan.inputs["metadata_selection"], project_root, "metadata selection"
    )
    try:
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != SELECTION_FIELDS:
                raise V4CalibrationMaterializationError(
                    "Calibration metadata CSV schema is invalid."
                )
            mappings = list(reader)
    except OSError as error:
        raise V4CalibrationMaterializationError(
            "Cannot read calibration metadata selection."
        ) from error
    if len(mappings) != plan.target_rows:
        raise V4CalibrationMaterializationError("Calibration metadata target row count changed.")
    rows: list[SelectionRow] = []
    values: dict[str, set[str]] = defaultdict(set)
    for rank, item in enumerate(mappings, start=1):
        try:
            row = SelectionRow(
                selection_rank=int(item.get("selection_rank") or ""),
                sample_id=(item.get("sample_id") or "").strip(),
                submission_pseudo_id=(item.get("submission_pseudo_id") or "").strip(),
                prompt_id=(item.get("prompt_id") or "").strip(),
                parent_group_id=(item.get("parent_group_id") or "").strip(),
                speaker_pseudo_id=(item.get("speaker_pseudo_id") or "").strip(),
                prompt_text_hash=_sha256(
                    item.get("prompt_text_hash"), "selection prompt text hash"
                ),
                original_prompt_text_hash=_sha256(
                    item.get("original_prompt_text_hash"), "selection original text hash"
                ),
            )
        except ValueError as error:
            raise V4CalibrationMaterializationError(
                f"Calibration selection row {rank + 1} is invalid."
            ) from error
        if (
            row.selection_rank != rank
            or row.parent_group_id != row.speaker_pseudo_id
            or not row.prompt_id
        ):
            raise V4CalibrationMaterializationError(
                "Calibration selection ranks or group binding changed."
            )
        for field in (
            "sample_id",
            "parent_group_id",
            "speaker_pseudo_id",
            "prompt_text_hash",
            "original_prompt_text_hash",
        ):
            value = getattr(row, field)
            if not value or value in values[field]:
                raise V4CalibrationMaterializationError(f"Calibration selection repeats {field}.")
            values[field].add(value)
        rows.append(row)
    return tuple(rows)


def _require_train_dev_isolation(
    rows: Sequence[SelectionRow], plan: CalibrationPlan, project_root: Path
) -> None:
    for name in ("v4_train_manifest", "v4_dev_manifest"):
        manifest = load_manifest(_verify_binding(plan.inputs[name], project_root, name))
        validate_manifest(manifest)
        for field, selected in {
            "sample_id": {row.sample_id for row in rows},
            "text_hash": {row.prompt_text_hash for row in rows},
            "parent_group_id": {row.parent_group_id for row in rows},
            "speaker_pseudo_id": {row.speaker_pseudo_id for row in rows},
        }.items():
            if selected.intersection(getattr(row, field) for row in manifest):
                raise V4CalibrationMaterializationError(
                    f"Calibration selection overlaps frozen {name} on {field}."
                )


def _require_source_route(
    plan: CalibrationPlan, project_root: Path
) -> tuple[ResearchTtsModel, Mapping[str, Path]]:
    source_audit = _json_object(
        _verify_binding(plan.inputs["voxforge_source_audit"], project_root, "VoxForge audit"),
        "VoxForge audit",
    )
    if (
        source_audit.get("source_id") != "voxforge_ru_mdc_2026_05"
        or source_audit.get("archive_sha256") != plan.archive_sha256
        or source_audit.get("archive_size_bytes") != plan.archive_size_bytes
        or source_audit.get("wav_files") != 6412
    ):
        raise V4CalibrationMaterializationError("Pinned VoxForge source audit changed.")
    ledger = load_license_ledger(
        _verify_binding(plan.inputs["frozen_license_ledger"], project_root, "frozen ledger")
    )
    for source_id in (SOURCE_ID, SPOOF_ID):
        entry = ledger.get(source_id)
        if (
            entry is None
            or entry.status not in APPROVED_LICENSE_STATUSES
            or entry.usage_scope != "personal_research"
        ):
            raise V4CalibrationMaterializationError(
                f"Frozen calibration ledger does not approve {source_id!r}."
            )
    lock = load_research_tts_model_lock(
        _verify_binding(plan.inputs["espeak_model_lock"], project_root, "eSpeak model lock")
    )
    if len(lock.models) != 1 or lock.models[0].model_id != plan.model_id:
        raise V4CalibrationMaterializationError(
            "Materialization must use exactly the pinned RU eSpeak model."
        )
    model = lock.models[0]
    runtime = load_espeakng_runtime(model)
    if runtime.voice != "ru" or model.generator_family != "formant_rule_based_tts":
        raise V4CalibrationMaterializationError(
            "Pinned synthesis route is not Russian eSpeak formant TTS."
        )
    verified = verify_research_tts_model_lock(
        _project_path(project_root, plan.model_root, "eSpeak model root"), lock
    )
    return model, verified[model.model_id]


def _bind_records(selection: Sequence[SelectionRow], archive: Path) -> tuple[BoundSelection, ...]:
    records = load_voxforge_ru_metadata(archive)
    indexed = {voxforge_metadata_identity(record).sample_id: record for record in records}
    if len(indexed) != len(records):
        raise V4CalibrationMaterializationError(
            "Pinned VoxForge archive has duplicate metadata identities."
        )
    bound: list[BoundSelection] = []
    for row in selection:
        record = indexed.get(row.sample_id)
        if record is None:
            raise V4CalibrationMaterializationError(
                f"Selected VoxForge identity vanished: {row.sample_id!r}."
            )
        identity = voxforge_metadata_identity(record)
        if (
            identity.parent_group_id != row.parent_group_id
            or identity.speaker_pseudo_id != row.speaker_pseudo_id
            or identity.prompt_text_hash != row.prompt_text_hash
            or identity.original_prompt_text_hash != row.original_prompt_text_hash
            or record.prompt_id != row.prompt_id
            or hashlib.sha256(record.submission_id.encode()).hexdigest() != row.submission_pseudo_id
        ):
            raise V4CalibrationMaterializationError(
                f"Pinned VoxForge metadata changed for {row.sample_id!r}."
            )
        bound.append(BoundSelection(selection=row, record=record))
    return tuple(bound)


def _safe_member_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise V4CalibrationMaterializationError("VoxForge TAR contains an unsafe member path.")
    return path


def extract_selected_wavs(
    archive: Path, bound: Sequence[BoundSelection], destination: Path
) -> dict[str, Path]:
    """Extract only frozen selected WAV bytes into a new Git-ignored namespace."""

    if destination.exists() or not destination.parent.is_dir():
        raise V4CalibrationMaterializationError(
            "Raw VoxForge destination is not a new existing-parent path."
        )
    wanted = {
        (
            f"{VOXFORGE_RU_ARCHIVE_ROOT}/{item.record.submission_id}/wav/"
            f"{item.record.prompt_id}.wav"
        ): item
        for item in bound
    }
    if len(wanted) != len(bound):
        raise V4CalibrationMaterializationError(
            "Selected VoxForge metadata maps multiple rows to one WAV."
        )
    stage = Path(tempfile.mkdtemp(prefix=".kds-v4-calibration-voxforge-", dir=destination.parent))
    staged_destination = stage / destination.name
    staged_destination.mkdir()
    extracted: dict[str, Path] = {}
    try:
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                _safe_member_path(member.name)
                if not (member.isdir() or member.isfile()):
                    raise V4CalibrationMaterializationError(
                        "VoxForge TAR contains an unsafe member type."
                    )
                item = wanted.get(member.name)
                if item is None:
                    continue
                if not member.isfile() or member.size <= 44:
                    raise V4CalibrationMaterializationError(
                        "Selected VoxForge member is not a WAV payload."
                    )
                target = staged_destination / f"voxforge_ru_{item.selection.selection_rank:03d}.wav"
                source = handle.extractfile(member)
                if source is None:
                    raise V4CalibrationMaterializationError("Cannot read selected VoxForge member.")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                with wave.open(str(target), "rb") as audio:
                    if audio.getnframes() <= 0 or audio.getframerate() != 48000:
                        raise V4CalibrationMaterializationError(
                            "Selected VoxForge WAV is invalid or not 48 kHz."
                        )
                extracted[item.selection.sample_id] = target
        if set(extracted) != {item.selection.sample_id for item in bound}:
            raise V4CalibrationMaterializationError("Pinned archive lacks a selected WAV.")
        staged_destination.replace(destination)
    except (OSError, tarfile.TarError, wave.Error) as error:
        raise V4CalibrationMaterializationError(
            "Cannot safely extract selected VoxForge WAVs."
        ) from error
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {sample: destination / asset.name for sample, asset in extracted.items()}


def _relative_to_data_root(data_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(data_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise V4CalibrationMaterializationError(
            "Audio asset path escapes the data root."
        ) from error


def _source_raw_rows(
    bound: Sequence[BoundSelection], extracted: Mapping[str, Path], data_root: Path, created_at: str
) -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    for item in bound:
        path = extracted[item.selection.sample_id]
        with wave.open(str(path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
            sample_rate = audio.getframerate()
        rows.append(
            ManifestRow(
                sample_id=item.selection.sample_id,
                relative_path=_relative_to_data_root(data_root, path),
                sha256=sha256_file(path),
                split="test",
                label="bonafide",
                language="ru",
                code_switch="unknown",
                parent_group_id=item.selection.parent_group_id,
                source_name=SOURCE_ID,
                source_license=VOXFORGE_RU_LICENSE,
                rights_basis=(
                    "Pinned VoxForge Russian GPL-3.0-or-later archive; one frozen local "
                    "personal-research calibration-preparation route only"
                ),
                speaker_pseudo_id=item.selection.speaker_pseudo_id,
                text_id=item.selection.prompt_id,
                text_hash=item.selection.prompt_text_hash,
                duration_s=duration,
                generator_family="",
                generator_name="",
                generator_version="",
                voice_id="",
                clone_consent_id="",
                device="unknown",
                capture_route="voxforge_submission_read_speech",
                original_sr=sample_rate,
                codec="wav",
                augmentation_chain="none",
                augmentation_seed="",
                created_at=created_at,
            )
        )
    validate_manifest(rows)
    return tuple(rows)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def synthesize_espeak_once(
    *,
    source_ready: Sequence[ManifestRow],
    text_by_sample: Mapping[str, str],
    rank_by_sample: Mapping[str, int],
    model: ResearchTtsModel,
    verified: Mapping[str, Path],
    data_root: Path,
    destination: Path,
    runtime_root: Path,
    created_at: str,
) -> tuple[ManifestRow, ...]:
    """Generate exactly one local text-only eSpeak WAV for each ready frozen source row."""

    journal = runtime_root / "espeak_one_shot.jsonl"
    if destination.exists() or not destination.parent.is_dir() or journal.exists():
        raise V4CalibrationMaterializationError(
            "eSpeak output/journal namespace is not new; resynthesis is forbidden."
        )
    destination.mkdir()
    runtime = load_espeakng_runtime(model)
    rows: list[ManifestRow] = []
    try:
        with tempfile.TemporaryDirectory(prefix="kds-v4-calibration-espeak-") as temp_name:
            paths = extract_verified_espeakng_runtime(
                verified, runtime, Path(temp_name) / "runtime"
            )
            for index, base in enumerate(
                sorted(source_ready, key=lambda row: rank_by_sample[row.sample_id]), start=1
            ):
                profile: EspeakNgProfile = runtime.profiles[(index - 1) % len(runtime.profiles)]
                key = hashlib.sha256(
                    f"{base.sample_id}:{model.model_id}:{profile.voice_id}".encode()
                ).hexdigest()[:20]
                output = destination / f"espeak_ru_{rank_by_sample[base.sample_id]:03d}_{key}.wav"
                _append_jsonl(
                    journal,
                    {
                        "event": "planned",
                        "base_sample_id": base.sample_id,
                        "profile": profile.voice_id,
                        "output": output.name,
                    },
                )
                synthesize_espeakng(
                    runtime_paths=paths,
                    runtime=runtime,
                    profile=profile,
                    text=text_by_sample[base.sample_id],
                    output=output,
                )
                info = sf.info(str(output))
                if (
                    info.duration <= 0
                    or info.samplerate != runtime.sample_rate
                    or str(info.format).lower() != "wav"
                ):
                    raise V4CalibrationMaterializationError(
                        "eSpeak generated an invalid locked WAV."
                    )
                sample_key = hashlib.sha256(
                    f"{base.sample_id}:{model.model_id}:{profile.voice_id}".encode()
                ).hexdigest()[:16]
                row = ManifestRow(
                    sample_id=f"{SPOOF_ID}:{sample_key}",
                    relative_path=_relative_to_data_root(data_root, output),
                    sha256=sha256_file(output),
                    split="test",
                    label="spoof",
                    language="ru",
                    code_switch="unknown",
                    parent_group_id=f"{SPOOF_ID}:text-only:{model.model_id}:{profile.voice_id}",
                    source_name=SPOOF_ID,
                    source_license=f"VoxForge GPL-3.0-or-later text; {model.license}",
                    rights_basis=(
                        "One-shot offline text-only derivative of frozen VoxForge prompt "
                        f"{base.text_id}; no reference audio, cloning, resynthesis or replacement"
                    ),
                    speaker_pseudo_id=f"{SPOOF_ID}:synthetic-control:{model.model_id}:{profile.voice_id}",
                    text_id=base.text_id,
                    text_hash=base.text_hash,
                    duration_s=float(info.duration),
                    generator_family=model.generator_family,
                    generator_name=model.generator_name,
                    generator_version=model.generator_version,
                    voice_id=f"{model.model_id}:{profile.voice_id}",
                    clone_consent_id="not_applicable:text_only_formant_tts_no_reference_audio",
                    device="local_cpu_espeakng_formant",
                    capture_route="offline_text_only_formant_tts",
                    original_sr=int(info.samplerate),
                    codec="wav",
                    augmentation_chain="none",
                    augmentation_seed="",
                    created_at=created_at,
                )
                rows.append(row)
                _append_jsonl(
                    journal,
                    {
                        "event": "generated",
                        "base_sample_id": base.sample_id,
                        "sample_id": row.sample_id,
                        "audio_sha256": row.sha256,
                        "profile": profile.voice_id,
                    },
                )
    except BaseException:
        # The destination and journal remain as evidence; automatic retry would violate
        # one-shot synthesis.
        raise
    if len(rows) != len(source_ready):
        raise V4CalibrationMaterializationError(
            "One-shot eSpeak output does not cover every ready source row."
        )
    validate_manifest(rows)
    return tuple(rows)


def _decoded_relative_path(raw_sha256: str, namespace: str) -> str:
    _sha256(raw_sha256, "raw audio SHA-256")
    return (
        "processed/v4/xlsr_sls_model_v4_calibration_materialization_v1/"
        f"{namespace}/{raw_sha256[:2]}/{raw_sha256}.wav"
    )


def _run_decode_tasks(
    tasks: Mapping[str, V4DecodeTask], journal: Path, workers: int, stage: str
) -> dict[str, V4DecodeResult]:
    results = load_v4_decode_journal(journal, tasks)
    pending = [task for sample_id, task in tasks.items() if sample_id not in results]
    if not pending:
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[V4DecodeResult], V4DecodeTask] = {
            executor.submit(run_v4_decode_task, task): task for task in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_v4_decode_journal(journal, result)
            results[result.sample_id] = result
            if completed % 20 == 0 or completed == len(pending):
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "stage": stage,
                            "completed": len(results),
                            "total": len(tasks),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return results


def _is_manifest_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4CalibrationMaterializationError(
            f"Cannot inspect manifest inventory {path}."
        ) from error
    return REQUIRED_FIELDS.issubset(fields)


def _add_signature(
    mapping: dict[str, V4AudioSignature], key: str, signature: V4AudioSignature
) -> None:
    prior = mapping.get(key)
    if prior is not None and prior != signature:
        raise V4CalibrationMaterializationError(
            "Historical audio hash has contradictory canonical fingerprint data."
        )
    mapping[key] = signature


def _history_from_current_manifests(
    plan: CalibrationPlan, project_root: Path
) -> tuple[dict[str, tuple[str, ...]], tuple[V4AudioSignature, ...], dict[str, object]]:
    """Rebind full *current* manifest history to frozen prior fingerprint evidence.

    The 392 old ML-DF items without local media remain exact-only references. Any other current
    asset missing known canonical fingerprint evidence stops the gate rather than shrinking the
    history silently.
    """

    manifest_root = project_root / "data/manifests"
    references_by_hash: dict[str, set[str]] = defaultdict(set)
    bindings: list[dict[str, object]] = []
    row_count = 0
    for path in sorted(manifest_root.rglob("*.csv")):
        if not _is_manifest_csv(path):
            continue
        rows = load_manifest(path)
        validate_manifest(rows)
        relative = path.relative_to(project_root).as_posix()
        bindings.append({"path": relative, "sha256": sha256_file(path), "rows": len(rows)})
        row_count += len(rows)
        for row in rows:
            references_by_hash[row.sha256].add(f"{relative}:{row.sample_id}")
    known: set[str] = set()
    signatures_by_hash: dict[str, V4AudioSignature] = {}
    baseline = _verify_binding(
        plan.inputs["historical_fingerprint_inventory"],
        project_root,
        "historical fingerprint inventory",
    )
    with baseline.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "manifest_audio_sha256",
            "canonical_audio_sha256",
            "speech_seconds",
            "audio_fingerprint_v1",
            "fingerprint_status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise V4CalibrationMaterializationError(
                "Historical fingerprint inventory schema is invalid."
            )
        for mapping in reader:
            raw_hash = _sha256(mapping["manifest_audio_sha256"], "historical manifest SHA-256")
            known.add(raw_hash)
            if mapping["fingerprint_status"] != "fingerprinted":
                continue
            canonical = _sha256(mapping["canonical_audio_sha256"], "historical canonical SHA-256")
            signature = V4AudioSignature(
                identity=f"history:{canonical}",
                audio_sha256=canonical,
                fingerprint=mapping["audio_fingerprint_v1"],
                speech_seconds=float(mapping["speech_seconds"]),
            )
            known.add(canonical)
            _add_signature(signatures_by_hash, raw_hash, signature)
            _add_signature(signatures_by_hash, canonical, signature)
    for name in ("source_decode_inventory", "kk_spoof_decode_inventory"):
        path = _verify_binding(plan.inputs[name], project_root, name)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "raw_audio_sha256",
                "decoded_audio_sha256",
                "audio_fingerprint_v1",
                "speech_seconds",
                "preparation_status",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise V4CalibrationMaterializationError(f"{name} schema is invalid.")
            for mapping in reader:
                raw_hash = _sha256(mapping["raw_audio_sha256"], f"{name} raw SHA-256")
                known.add(raw_hash)
                if not mapping["decoded_audio_sha256"]:
                    continue
                canonical = _sha256(mapping["decoded_audio_sha256"], f"{name} decoded SHA-256")
                known.add(canonical)
                signature = V4AudioSignature(
                    identity=f"history:{canonical}",
                    audio_sha256=canonical,
                    fingerprint=mapping["audio_fingerprint_v1"],
                    speech_seconds=float(mapping["speech_seconds"]),
                )
                _add_signature(signatures_by_hash, raw_hash, signature)
                _add_signature(signatures_by_hash, canonical, signature)
    for name in ("dev_source_decode_journal", "dev_spoof_decode_journal"):
        path = _verify_binding(plan.inputs[name], project_root, name)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                mapping = cast(dict[str, object], json.loads(line))
                raw_hash = _sha256(mapping["raw_sha256"], f"{name} raw SHA-256")
            except (TypeError, ValueError, json.JSONDecodeError, KeyError) as error:
                raise V4CalibrationMaterializationError(
                    f"{name} line {line_number} is invalid."
                ) from error
            known.add(raw_hash)
            decoded = mapping.get("decoded_audio_sha256")
            if not isinstance(decoded, str) or not decoded:
                continue
            canonical = _sha256(decoded, f"{name} decoded SHA-256")
            known.add(canonical)
            fingerprint = mapping.get("audio_fingerprint_v1")
            speech_seconds = mapping.get("speech_seconds")
            if not isinstance(fingerprint, str) or not isinstance(speech_seconds, (int, float)):
                raise V4CalibrationMaterializationError(f"{name} fingerprint row is invalid.")
            signature = V4AudioSignature(
                identity=f"history:{canonical}",
                audio_sha256=canonical,
                fingerprint=fingerprint,
                speech_seconds=float(speech_seconds),
            )
            _add_signature(signatures_by_hash, raw_hash, signature)
            _add_signature(signatures_by_hash, canonical, signature)
    missing = set(references_by_hash).difference(known)
    if missing:
        raise V4CalibrationMaterializationError(
            "Current history has "
            f"{len(missing)} audio hashes without frozen exact/fingerprint coverage."
        )
    exact = {
        audio_hash: tuple(sorted(references))
        for audio_hash, references in references_by_hash.items()
    }
    canonical_signatures = {
        signature.identity: signature for signature in signatures_by_hash.values()
    }
    exact_only = len(set(references_by_hash).difference(signatures_by_hash))
    return (
        exact,
        tuple(canonical_signatures.values()),
        {
            "manifest_files": bindings,
            "manifest_rows": row_count,
            "unique_audio_hashes": len(references_by_hash),
            "near_fingerprint_covered_hashes": len(
                set(references_by_hash).intersection(signatures_by_hash)
            ),
            "exact_only_hashes": exact_only,
        },
    )


def _raw_exact_matches(
    decision: V4DecodedDecision, historical: Mapping[str, Sequence[str]]
) -> V4DecodedDecision:
    raw_matches = tuple(sorted(historical.get(decision.candidate.result.raw_sha256, ())))
    if not raw_matches:
        return decision
    return replace(
        decision, eligibility_status="rejected", rejection_reason="historical_exact_raw_audio"
    )


def _ready_rows(
    raw_rows: Mapping[str, ManifestRow], decisions: Sequence[V4DecodedDecision], created_at: str
) -> tuple[ManifestRow, ...]:
    ready: list[ManifestRow] = []
    for decision in decisions:
        if decision.eligibility_status != "eligible":
            continue
        raw = raw_rows[decision.candidate.result.sample_id]
        result = decision.candidate.result
        ready.append(
            replace(
                raw,
                relative_path=result.decoded_relative_path,
                sha256=result.decoded_audio_sha256,
                duration_s=result.duration_s,
                original_sr=16000,
                codec="wav",
                created_at=created_at,
            )
        )
    validate_manifest(ready)
    return tuple(ready)


def _inventory_rows(
    kind: str,
    ranks: Mapping[str, int],
    pair_map: Mapping[str, str],
    decisions: Sequence[V4DecodedDecision],
    historical: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for decision in decisions:
        result = decision.candidate.result
        rows.append(
            {
                "kind": kind,
                "selection_rank": ranks[result.sample_id],
                "sample_id": result.sample_id,
                "paired_source_sample_id": pair_map.get(result.sample_id, ""),
                "text_id": "",
                "text_hash": "",
                "raw_relative_path": result.raw_relative_path,
                "raw_audio_sha256": result.raw_sha256,
                "decoded_relative_path": result.decoded_relative_path,
                "decoded_audio_sha256": result.decoded_audio_sha256,
                "preparation_status": result.preparation_status,
                "eligibility_status": decision.eligibility_status,
                "rejection_reason": decision.rejection_reason,
                "historical_raw_exact_match_count": len(historical.get(result.raw_sha256, ())),
                "historical_decoded_exact_match_count": len(decision.historical_exact_matches),
                "historical_near_match_count": len(decision.historical_near_matches),
                "within_pool_near_match_count": len(decision.within_pool_near_matches),
            }
        )
    return rows


def _pair_lock(
    source_ready: Sequence[ManifestRow],
    spoof_ready: Sequence[ManifestRow],
    rank_by_source: Mapping[str, int],
) -> tuple[ManifestRow, ...]:
    source_by_text = {row.text_hash: row for row in source_ready}
    spoof_by_text = {row.text_hash: row for row in spoof_ready}
    if len(source_by_text) != len(source_ready) or len(spoof_by_text) != len(spoof_ready):
        raise V4CalibrationMaterializationError("Ready calibration layer repeats a text hash.")
    pairs = sorted(
        set(source_by_text).intersection(spoof_by_text),
        key=lambda key: rank_by_source[source_by_text[key].sample_id],
    )
    if not pairs:
        raise V4CalibrationMaterializationError(
            "No complete source/eSpeak calibration pair survived the gate."
        )
    locked = tuple(
        item
        for text_hash in pairs
        for item in (source_by_text[text_hash], spoof_by_text[text_hash])
    )
    validate_manifest(locked)
    return locked


def _publish(
    *,
    plan: CalibrationPlan,
    project_root: Path,
    data_root: Path,
    source_raw: Sequence[ManifestRow],
    source_ready: Sequence[ManifestRow],
    spoof_raw: Sequence[ManifestRow],
    spoof_ready: Sequence[ManifestRow],
    decisions: Sequence[V4DecodedDecision],
    source_decisions: Sequence[V4DecodedDecision],
    history: Mapping[str, Sequence[str]],
    history_scope: Mapping[str, object],
    pair_lock: Sequence[ManifestRow],
    created_at: str,
) -> None:
    outputs = {
        name: _project_path(project_root, relative, f"output {name}")
        for name, relative in plan.outputs.items()
    }
    if any(path.exists() or not path.parent.is_dir() for path in outputs.values()):
        raise V4CalibrationMaterializationError(
            "Materialization metadata outputs are not all new existing-parent paths."
        )
    ledger = load_license_ledger(
        _verify_binding(plan.inputs["frozen_license_ledger"], project_root, "frozen ledger")
    )
    for rows in (source_raw, source_ready, spoof_raw, spoof_ready, pair_lock):
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
    for rows in (source_raw, source_ready, spoof_raw, spoof_ready, pair_lock):
        require_valid_assets(rows, data_root)
    source_rank = {
        row.sample_id: index
        for index, row in enumerate(sorted(source_raw, key=lambda row: row.sample_id), start=1)
    }
    # Ranks are overwritten below from source IDs so inventories never depend on lexical ordering.
    source_rank = {
        row.sample_id: int(row.relative_path.rsplit("_", 1)[-1].split(".", 1)[0])
        for row in source_raw
    }
    spoof_pair_map = {
        row.sample_id: next(
            source.sample_id for source in source_raw if source.text_hash == row.text_hash
        )
        for row in spoof_raw
    }
    inventory = _inventory_rows("source", source_rank, {}, source_decisions, history)
    inventory.extend(
        _inventory_rows(
            "spoof",
            {row.sample_id: source_rank[spoof_pair_map[row.sample_id]] for row in spoof_raw},
            spoof_pair_map,
            decisions,
            history,
        )
    )
    text_by_sample = {
        row.sample_id: (row.text_id, row.text_hash) for row in (*source_raw, *spoof_raw)
    }
    for item in inventory:
        item["text_id"], item["text_hash"] = text_by_sample[str(item["sample_id"])]
    runtime_root = _project_path(project_root, plan.runtime_root, "runtime root")
    journals = {
        "source_decode_journal": runtime_root / "source_decode_qa.jsonl",
        "spoof_decode_journal": runtime_root / "spoof_decode_qa.jsonl",
        "espeak_one_shot_journal": runtime_root / "espeak_one_shot.jsonl",
    }
    if any(not path.is_file() for path in journals.values()):
        raise V4CalibrationMaterializationError(
            "One-shot materialization runtime journal is missing."
        )
    staged_outputs: dict[str, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-calibration-materialization-", dir=project_root
        ) as stage_name:
            stage = Path(stage_name)
            staged_outputs = {name: stage / path.name for name, path in outputs.items()}
            write_manifest(staged_outputs["source_raw_manifest"], source_raw)
            write_manifest(staged_outputs["source_ready_manifest"], source_ready)
            write_manifest(staged_outputs["spoof_raw_manifest"], spoof_raw)
            write_manifest(staged_outputs["spoof_ready_manifest"], spoof_ready)
            write_manifest(staged_outputs["pair_lock_manifest"], pair_lock)
            with staged_outputs["audio_gate_inventory"].open(
                "x", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(
                    sorted(
                        inventory, key=lambda row: (str(row["kind"]), int(row["selection_rank"]))
                    )
                )
            receipt = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "created_at": created_at,
                "state": "ru_calibration_pairs_frozen_checkpoint_scoring_contract_required",
                "bindings": {
                    "plan": {"path": plan.path, "sha256": plan.sha256},
                    **{name: asdict(binding) for name, binding in sorted(plan.inputs.items())},
                    "runtime": {
                        name: {
                            "path": path.relative_to(project_root).as_posix(),
                            "sha256": sha256_file(path),
                            "rows": len(path.read_text(encoding="utf-8").splitlines()),
                        }
                        for name, path in journals.items()
                    },
                },
                "current_history_scope": history_scope,
                "sources": {
                    "voxforge": {
                        "archive_name": plan.archive_name,
                        "size_bytes": plan.archive_size_bytes,
                        "sha256": plan.archive_sha256,
                        "raw_source_id": SOURCE_ID,
                    },
                    "espeak": {"model_id": plan.model_id, "synthetic_source_id": SPOOF_ID},
                },
                "counts": {
                    "metadata_selected": plan.target_rows,
                    "source_raw": len(source_raw),
                    "source_ready": len(source_ready),
                    "spoof_raw_one_shot": len(spoof_raw),
                    "spoof_ready": len(spoof_ready),
                    "complete_pairs": len(pair_lock) // 2,
                    "pair_assets": len(pair_lock),
                },
                "audio_gate": {
                    "canonical_decode": "ffmpeg mono pcm_s16le 16000 Hz",
                    "technical_decode_qa_vad_performed": True,
                    "historical_raw_exact_screen_performed": True,
                    "historical_decoded_exact_screen_performed": True,
                    "historical_near_audio_screen_performed": True,
                    "within_pool_exact_and_near_audio_screen_performed": True,
                    "source_decision_counts": dict(
                        sorted(
                            Counter(item.eligibility_status for item in source_decisions).items()
                        )
                    ),
                    "spoof_decision_counts": dict(
                        sorted(Counter(item.eligibility_status for item in decisions).items())
                    ),
                },
                "outputs": {
                    name: {
                        "path": plan.outputs[name],
                        "sha256": sha256_file(path),
                        "rows": _csv_rows(path),
                    }
                    for name, path in staged_outputs.items()
                    if name != "receipt"
                },
                "claims": {
                    "archive_rebinding_performed": True,
                    "raw_audio_extraction_performed": True,
                    "synthetic_audio_generated": True,
                    "exactly_one_espeak_attempt_per_ready_source_text": True,
                    "technical_decode_qa_vad_performed": True,
                    "complete_pair_lock_performed": True,
                    "checkpoint_loaded": False,
                    "calibration_performed": False,
                    "temperature_fitted": False,
                    "final_inference_performed": False,
                    "detector_inference_performed": False,
                    "speaker_independence": "not_verified_speaker_independent",
                    "calibration_language_scope": "ru_only",
                    "kk_probability_claim": False,
                    "replacement_or_backfill": False,
                    "resynthesis": False,
                },
                "next_gate": (
                    "separate checkpoint-scoring-and-calibration contract binding this pair lock; "
                    "no fitting or inference is authorized by this receipt"
                ),
            }
            staged_outputs["receipt"].write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for name, output in outputs.items():
                os.link(staged_outputs[name], output)
                published.append((output, staged_outputs[name]))
    except (OSError, ValueError) as error:
        for output, staged in reversed(published):
            try:
                if output.samefile(staged):
                    output.unlink()
            except OSError:
                pass
        raise V4CalibrationMaterializationError(
            "Cannot publish materialization receipt packet atomically."
        ) from error


def run_calibration_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    voxforge_archive: Path,
    workers: int,
    created_at: str,
) -> CalibrationPlan:
    """Execute the isolated materialization gate, never loading a detector checkpoint."""

    if workers not in range(1, 65):
        raise V4CalibrationMaterializationError("Decode workers must be between 1 and 64.")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4CalibrationMaterializationError("Materialization timestamp is invalid.") from error
    root = project_root.resolve(strict=True)
    data = data_root.resolve(strict=True)
    plan = load_calibration_materialization_plan(plan_path, root)
    archive = voxforge_archive.resolve(strict=True)
    if (
        archive.name != plan.archive_name
        or archive.stat().st_size != plan.archive_size_bytes
        or sha256_file(archive) != plan.archive_sha256
    ):
        raise V4CalibrationMaterializationError(
            "VoxForge archive does not match the materialization contract."
        )
    output_paths = [
        _project_path(root, value, f"output {name}") for name, value in plan.outputs.items()
    ]
    roots = [
        _project_path(root, value, name)
        for name, value in {
            "raw source": plan.raw_source_root,
            "raw spoof": plan.raw_spoof_root,
            "processed source": plan.processed_source_root,
            "processed spoof": plan.processed_spoof_root,
            "runtime": plan.runtime_root,
        }.items()
    ]
    if any(path.exists() for path in (*output_paths, *roots)):
        raise V4CalibrationMaterializationError(
            "Write-once materialization output or working namespace already exists."
        )
    if any(not path.parent.is_dir() for path in output_paths):
        raise V4CalibrationMaterializationError(
            "Materialization metadata output parent directory is missing."
        )
    selection = _load_selection(plan, root)
    _require_train_dev_isolation(selection, plan, root)
    model, verified = _require_source_route(plan, root)
    bound = _bind_records(selection, archive)
    history_exact, history_signatures, history_scope = _history_from_current_manifests(plan, root)
    raw_source_root = _project_path(root, plan.raw_source_root, "raw source root")
    try:
        raw_source_root.parent.mkdir()
    except OSError as error:
        raise V4CalibrationMaterializationError(
            "Cannot create the new dedicated raw calibration namespace."
        ) from error
    extracted = extract_selected_wavs(archive, bound, raw_source_root)
    source_raw = _source_raw_rows(bound, extracted, data, created_at)
    ledger = load_license_ledger(
        _verify_binding(plan.inputs["frozen_license_ledger"], root, "frozen ledger")
    )
    validate_manifest_licenses(source_raw, ledger)
    require_valid_assets(source_raw, data)
    runtime_root = _project_path(root, plan.runtime_root, "runtime root")
    runtime_root.mkdir()
    source_tasks = {
        row.sample_id: V4DecodeTask(
            sample_id=row.sample_id,
            raw_relative_path=row.relative_path,
            raw_sha256=row.sha256,
            source_path=str(resolve_asset_path(data, row.relative_path)),
            decoded_relative_path=_decoded_relative_path(row.sha256, "source"),
            destination_path=str(
                resolve_asset_path(data, _decoded_relative_path(row.sha256, "source"))
            ),
        )
        for row in source_raw
    }
    source_results = _run_decode_tasks(
        source_tasks,
        runtime_root / "source_decode_qa.jsonl",
        workers,
        "v4_calibration_source_decode_qa",
    )
    source_ranks = {item.selection.sample_id: item.selection.selection_rank for item in bound}
    source_decisions = tuple(
        _raw_exact_matches(item, history_exact)
        for item in decide_v4_decoded_audio_eligibility(
            tuple(
                V4DecodedCandidate(
                    selection_rank=source_ranks[sample_id],
                    language="ru",
                    label="bonafide",
                    result=result,
                )
                for sample_id, result in source_results.items()
            ),
            history_exact,
            history_signatures,
        )
    )
    source_ready = _ready_rows(
        {row.sample_id: row for row in source_raw}, source_decisions, created_at
    )
    if not source_ready:
        raise V4CalibrationMaterializationError(
            "No VoxForge source row survived the audio isolation gate."
        )
    text_by_sample = {item.selection.sample_id: item.record.prompt_text for item in bound}
    raw_spoof_root = _project_path(root, plan.raw_spoof_root, "raw spoof root")
    spoof_raw = synthesize_espeak_once(
        source_ready=source_ready,
        text_by_sample=text_by_sample,
        rank_by_sample=source_ranks,
        model=model,
        verified=verified,
        data_root=data,
        destination=raw_spoof_root,
        runtime_root=runtime_root,
        created_at=created_at,
    )
    validate_manifest_licenses(spoof_raw, ledger)
    require_valid_assets(spoof_raw, data)
    all_source_signatures = tuple(
        V4AudioSignature(
            identity=f"source:{result.sample_id}",
            audio_sha256=result.decoded_audio_sha256,
            fingerprint=result.audio_fingerprint_v1,
            speech_seconds=result.speech_seconds,
        )
        for result in source_results.values()
        if result.decoded_audio_sha256
    )
    synthetic_history_exact = {key: tuple(value) for key, value in history_exact.items()}
    for result in source_results.values():
        if result.decoded_audio_sha256:
            synthetic_history_exact[result.raw_sha256] = tuple(
                sorted(
                    {
                        *synthetic_history_exact.get(result.raw_sha256, ()),
                        f"source:{result.sample_id}",
                    }
                )
            )
            synthetic_history_exact[result.decoded_audio_sha256] = tuple(
                sorted(
                    {
                        *synthetic_history_exact.get(result.decoded_audio_sha256, ()),
                        f"source:{result.sample_id}",
                    }
                )
            )
    spoof_to_source = {
        row.sample_id: next(
            source.sample_id for source in source_ready if source.text_hash == row.text_hash
        )
        for row in spoof_raw
    }
    spoof_tasks = {
        row.sample_id: V4DecodeTask(
            sample_id=row.sample_id,
            raw_relative_path=row.relative_path,
            raw_sha256=row.sha256,
            source_path=str(resolve_asset_path(data, row.relative_path)),
            decoded_relative_path=_decoded_relative_path(row.sha256, "spoof"),
            destination_path=str(
                resolve_asset_path(data, _decoded_relative_path(row.sha256, "spoof"))
            ),
        )
        for row in spoof_raw
    }
    spoof_results = _run_decode_tasks(
        spoof_tasks,
        runtime_root / "spoof_decode_qa.jsonl",
        workers,
        "v4_calibration_spoof_decode_qa",
    )
    spoof_decisions = tuple(
        _raw_exact_matches(item, synthetic_history_exact)
        for item in decide_v4_decoded_audio_eligibility(
            tuple(
                V4DecodedCandidate(
                    selection_rank=source_ranks[spoof_to_source[sample_id]],
                    language="ru",
                    label="spoof",
                    result=result,
                )
                for sample_id, result in spoof_results.items()
            ),
            synthetic_history_exact,
            (*history_signatures, *all_source_signatures),
        )
    )
    spoof_ready = _ready_rows(
        {row.sample_id: row for row in spoof_raw}, spoof_decisions, created_at
    )
    pair_lock = _pair_lock(source_ready, spoof_ready, source_ranks)
    _publish(
        plan=plan,
        project_root=root,
        data_root=data,
        source_raw=source_raw,
        source_ready=source_ready,
        spoof_raw=spoof_raw,
        spoof_ready=spoof_ready,
        decisions=spoof_decisions,
        source_decisions=source_decisions,
        history=history_exact,
        history_scope=history_scope,
        pair_lock=pair_lock,
        created_at=created_at,
    )
    return plan
