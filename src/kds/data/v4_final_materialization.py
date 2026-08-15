"""Write-once materialization and review gate for the frozen v4 final inputs.

This module deliberately stops before checkpoint loading or detector inference.  It
can materialize only the 500+500 identities in the preceding metadata contract,
perform canonical decode/QA/VAD and full-history audio isolation, and publish an
exact-byte review packet.  A separate command can make a pair lock only after two
independent reviewers have approved every asset in a retained pair.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, resolve_asset_path, sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_SOURCE_LICENSE,
    CommonVoiceRecord,
    ExtractedCommonVoiceAsset,
    common_voice_manifest_rows,
    extract_common_voice_audio_slice,
    inspect_extracted_common_voice_audio,
    load_common_voice_metadata_from_archive,
)
from kds.data.fleurs import (
    FLEURS_LICENSE,
    FleursExtractedAsset,
    FleursRecord,
    extract_fleurs_audio_slice,
    fleurs_manifest_rows,
    inspect_extracted_fleurs_audio,
    inspect_fleurs_release,
)
from kds.data.kazakhtts import (
    extract_verified_kazakhtts_runtime,
    load_kazakhtts_runtime,
    validate_kazakhtts_text,
)
from kds.data.kazakhtts_inference import (
    load_kazakhtts_models,
    resolve_kazakhtts_device,
    synthesize_kazakhtts_waveform,
)
from kds.data.kazakhtts_text import (
    KAZAKHTTS_TEXT_NORMALIZER_ID,
    normalize_kazakhtts_stage_c_text,
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
from kds.data.qwen3_tts_customvoice import load_qwen3_tts_customvoice
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
from kds.data.v4_final_inputs import V4_FINAL_SELECTION_FIELDS

PROTOCOL_ID = "xlsr-sls-model-v4-final-materialization-v1"
PLAN_SCHEMA_VERSION = 1
SELECTION_PROTOCOL_ID = "xlsr-sls-model-v4-final-inputs-v1"
RU_SOURCE_ID = "common_voice_ru_v24_v4_final"
KK_SOURCE_ID = "google_fleurs_kk_v1_v4_final"
RU_SPOOF_ID = "qwen3_tts_customvoice_aiden_v4_final"
KK_SPOOF_ID = "issai_kazakhtts2_male2_tacotron2_pwg_v4_final"
_HEX = frozenset("0123456789abcdef")
_EXPECTED_RAW_ROOTS = {
    "ru_source": "raw/v4/xlsr_sls_model_v4_final_materialization_v1/ru_source",
    "kk_source": "raw/v4/xlsr_sls_model_v4_final_materialization_v1/kk_source",
    "ru_spoof": "raw/v4/xlsr_sls_model_v4_final_materialization_v1/ru_qwen",
    "kk_spoof": "raw/v4/xlsr_sls_model_v4_final_materialization_v1/kk_kazakhtts",
}
_EXPECTED_PROCESSED_ROOT = "processed/v4/xlsr_sls_model_v4_final_materialization_v1"
_EXPECTED_RUNTIME_ROOT = "artifacts/v4/xlsr_sls_model_v4_final_materialization_v1"

INVENTORY_FIELDS = (
    "selection_rank",
    "sample_id",
    "pair_key",
    "language",
    "label",
    "raw_relative_path",
    "raw_audio_sha256",
    "decoded_relative_path",
    "decoded_audio_sha256",
    "duration_s",
    "peak",
    "rms_dbfs",
    "clipped_fraction",
    "dc_offset",
    "speech_seconds",
    "speech_segment_count",
    "audio_fingerprint_v1",
    "preparation_status",
    "eligibility_status",
    "rejection_reason",
    "historical_raw_exact_match_count",
    "historical_decoded_exact_match_count",
    "historical_near_match_count",
    "within_pool_near_match_count",
)
PACKET_FIELDS = (
    "protocol_id",
    "materialization_receipt_sha256",
    "sample_id",
    "pair_key",
    "language",
    "label",
    "text_id",
    "text_hash",
    "audio_path",
    "audio_sha256",
    "source_text",
    "synthesis_text",
    "synthesis_text_sha256",
)
REVIEW_FIELDS = (
    "protocol_id",
    "packet_sha256",
    "sample_id",
    "audio_sha256",
    "reviewer_pseudo_id",
    "review_status",
    "speech_intelligible",
    "lexical_content_preserved",
    "language_preserved",
    "severe_artifacts",
    "notes",
)


class V4FinalMaterializationError(ValueError):
    """Raised when a final materialization boundary cannot be proven."""


@dataclass(frozen=True, slots=True)
class Binding:
    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class Plan:
    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, Binding]
    raw_roots: Mapping[str, str]
    processed_root: str
    runtime_root: str
    model_roots: Mapping[str, str]
    outputs: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SelectedRow:
    language: str
    selection_rank: int
    sample_id: str
    source_member: str
    source_split: str
    parent_group_id: str
    speaker_pseudo_id: str
    text_id: str
    text_hash: str
    synthesis_text_sha256: str
    synthesis_seed: str
    normalization_operations: str

    @property
    def pair_key(self) -> str:
        return f"{self.language}:{self.selection_rank:03d}:{self.text_hash}"


@dataclass(frozen=True, slots=True)
class ReviewPacketRow:
    protocol_id: str
    materialization_receipt_sha256: str
    sample_id: str
    pair_key: str
    language: str
    label: str
    text_id: str
    text_hash: str
    audio_path: str
    audio_sha256: str
    source_text: str
    synthesis_text: str
    synthesis_text_sha256: str


@dataclass(frozen=True, slots=True)
class ReviewRow:
    protocol_id: str
    packet_sha256: str
    sample_id: str
    audio_sha256: str
    reviewer_pseudo_id: str
    review_status: str
    speech_intelligible: str
    lexical_content_preserved: str
    language_preserved: str
    severe_artifacts: str
    notes: str


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalMaterializationError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4FinalMaterializationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4FinalMaterializationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4FinalMaterializationError(f"{label} must be a non-empty project-relative path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise V4FinalMaterializationError(f"{label} is not a safe project-relative path.")
    return parsed.as_posix()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(item not in _HEX for item in value):
        raise V4FinalMaterializationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4FinalMaterializationError(f"{label} must be an ISO-8601 timestamp.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4FinalMaterializationError(f"{label} must be an ISO-8601 timestamp.") from error
    return value


def _positive(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4FinalMaterializationError(f"{label} must be a positive integer.")
    return value


def _binding(value: object, label: str) -> Binding:
    raw = _mapping(value, label)
    if set(raw) != {"path", "sha256", "rows"}:
        raise V4FinalMaterializationError(f"{label} must contain path, sha256 and rows.")
    rows = raw["rows"]
    if rows is not None:
        _positive(rows, f"{label}.rows")
    return Binding(
        _safe_path(raw["path"], f"{label}.path"),
        _sha(raw["sha256"], f"{label}.sha256"),
        cast(int | None, rows),
    )


def _project_path(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise V4FinalMaterializationError(f"{label} escapes project root.") from error
    return path


def _rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4FinalMaterializationError(f"Cannot count CSV rows: {path}") from error


def _verify(binding: Binding, root: Path, label: str) -> Path:
    path = _project_path(root, binding.path, label)
    if not path.is_file() or sha256_file(path) != binding.sha256:
        raise V4FinalMaterializationError(f"{label} binding does not match: {binding.path}")
    if binding.rows is not None and _rows(path) != binding.rows:
        raise V4FinalMaterializationError(f"{label} row count changed: {binding.path}")
    return path


def load_plan(path: Path, project_root: Path) -> Plan:
    """Load the narrow, hash-pinned authorization for final materialization only."""

    root = project_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise V4FinalMaterializationError("Plan must be below project root.") from error
    raw = _object(resolved, "v4 final materialization plan")
    expected = {
        "schema_version",
        "protocol_id",
        "created_at",
        "inputs",
        "working",
        "outputs",
        "prohibitions",
    }
    if (
        set(raw) != expected
        or raw["schema_version"] != PLAN_SCHEMA_VERSION
        or raw["protocol_id"] != PROTOCOL_ID
    ):
        raise V4FinalMaterializationError("Plan schema/protocol is invalid.")
    input_values = _mapping(raw["inputs"], "inputs")
    required_inputs = {
        "metadata_plan",
        "metadata_receipt",
        "metadata_selection",
        "materialization_ledger",
        "fleurs_artifact_lock",
        "qwen_model_lock",
        "kazakhtts_model_lock",
        "train_manifest",
        "dev_manifest",
        "historical_fingerprint_inventory",
        "source_decode_inventory",
        "kk_spoof_decode_inventory",
        "dev_source_decode_journal",
        "dev_spoof_decode_journal",
        "final_inputs_module",
        "materialization_module",
        "audio_gate_module",
        "common_voice_module",
        "fleurs_module",
        "qwen_module",
        "kazakhtts_module",
        "kazakhtts_inference_module",
        "runner_script",
    }
    if set(input_values) != required_inputs:
        raise V4FinalMaterializationError("Plan inputs are incomplete.")
    inputs = {name: _binding(value, f"inputs.{name}") for name, value in input_values.items()}
    working = _mapping(raw["working"], "working")
    if set(working) != {"raw_roots", "processed_root", "runtime_root", "model_roots"}:
        raise V4FinalMaterializationError("Plan working section is invalid.")
    raw_roots_value = _mapping(working["raw_roots"], "working.raw_roots")
    if set(raw_roots_value) != {"ru_source", "kk_source", "ru_spoof", "kk_spoof"}:
        raise V4FinalMaterializationError("Plan raw roots are invalid.")
    raw_roots = {
        name: _safe_path(value, f"working.raw_roots.{name}")
        for name, value in raw_roots_value.items()
    }
    model_roots_value = _mapping(working["model_roots"], "working.model_roots")
    if set(model_roots_value) != {"qwen", "kazakhtts"}:
        raise V4FinalMaterializationError("Plan model roots are invalid.")
    model_roots = {
        name: _safe_path(value, f"working.model_roots.{name}")
        for name, value in model_roots_value.items()
    }
    processed_root = _safe_path(working["processed_root"], "working.processed_root")
    runtime_root = _safe_path(working["runtime_root"], "working.runtime_root")
    if (
        raw_roots != _EXPECTED_RAW_ROOTS
        or processed_root != _EXPECTED_PROCESSED_ROOT
        or runtime_root != _EXPECTED_RUNTIME_ROOT
    ):
        raise V4FinalMaterializationError(
            "Plan must retain the fixed one-shot materialization namespaces."
        )
    all_roots = [*raw_roots.values(), processed_root, runtime_root]
    if len(all_roots) != len(set(all_roots)):
        raise V4FinalMaterializationError("Working output roots must be distinct.")
    outputs_value = _mapping(raw["outputs"], "outputs")
    required_outputs = {
        "ru_source_raw_manifest",
        "kk_source_raw_manifest",
        "ru_spoof_raw_manifest",
        "kk_spoof_raw_manifest",
        "ru_source_ready_manifest",
        "kk_source_ready_manifest",
        "ru_spoof_ready_manifest",
        "kk_spoof_ready_manifest",
        "audio_inventory",
        "review_packet",
        "reviewer_a_template",
        "reviewer_b_template",
        "materialization_receipt",
        "pair_lock_manifest",
        "pair_lock_receipt",
    }
    if set(outputs_value) != required_outputs:
        raise V4FinalMaterializationError("Plan output set is invalid.")
    outputs = {name: _safe_path(value, f"outputs.{name}") for name, value in outputs_value.items()}
    if len(set(outputs.values())) != len(outputs):
        raise V4FinalMaterializationError("Plan outputs must be distinct.")
    prohibitions = _mapping(raw["prohibitions"], "prohibitions")
    expected_prohibitions = {
        "network_downloads",
        "detector_checkpoint_loading",
        "calibration",
        "temperature_fitting",
        "final_inference",
        "detector_inference",
        "detector_feedback",
        "output_overwrite",
        "resynthesis",
        "replacement_or_backfill",
        "pair_lock_before_two_reviews",
    }
    if set(prohibitions) != expected_prohibitions or any(
        value is not True for value in prohibitions.values()
    ):
        raise V4FinalMaterializationError("Plan prohibitions are not fail-closed.")
    for name, binding in inputs.items():
        _verify(binding, root, f"inputs.{name}")
    return Plan(
        relative,
        sha256_file(resolved),
        _timestamp(raw["created_at"], "created_at"),
        inputs,
        raw_roots,
        processed_root,
        runtime_root,
        model_roots,
        outputs,
    )


def _load_selection(plan: Plan, root: Path) -> tuple[SelectedRow, ...]:
    receipt = _object(
        _verify(plan.inputs["metadata_receipt"], root, "metadata receipt"), "metadata receipt"
    )
    claims = _mapping(receipt.get("claims"), "metadata receipt claims")
    receipt_plan = _mapping(receipt.get("plan"), "metadata receipt plan")
    selection_claims = _mapping(receipt.get("selection"), "metadata receipt selection")
    ru_claims = _mapping(selection_claims.get("ru"), "metadata receipt RU selection")
    kk_claims = _mapping(selection_claims.get("kk"), "metadata receipt KK selection")
    if (
        receipt.get("protocol_id") != SELECTION_PROTOCOL_ID
        or receipt.get("status") != "ok"
        or receipt_plan.get("path") != plan.inputs["metadata_plan"].path
        or receipt_plan.get("sha256") != plan.inputs["metadata_plan"].sha256
        or ru_claims.get("selected_pairs") != 500
        or kk_claims.get("selected_pairs") != 500
        or claims.get("raw_audio_extraction_performed") is not False
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("audio_qa_performed") is not False
        or claims.get("acoustic_review_performed") is not False
        or claims.get("pairing_performed") is not False
        or claims.get("final_inference_performed") is not False
        or claims.get("future_materialization_requires_separate_contract") is not True
    ):
        raise V4FinalMaterializationError(
            "Metadata receipt does not authorize this contract boundary."
        )
    path = _verify(plan.inputs["metadata_selection"], root, "metadata selection")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != V4_FINAL_SELECTION_FIELDS:
                raise V4FinalMaterializationError("Final metadata selection schema changed.")
            source = list(reader)
    except OSError as error:
        raise V4FinalMaterializationError("Cannot read final metadata selection.") from error
    selected: list[SelectedRow] = []
    seen: dict[str, set[str]] = defaultdict(set)
    for number, item in enumerate(source, start=2):
        try:
            row = SelectedRow(
                language=(item.get("language") or "").strip(),
                selection_rank=int(item.get("selection_rank") or ""),
                sample_id=(item.get("sample_id") or "").strip(),
                source_member=(item.get("source_member") or "").strip(),
                source_split=(item.get("source_split") or "").strip(),
                parent_group_id=(item.get("parent_group_id") or "").strip(),
                speaker_pseudo_id=(item.get("speaker_pseudo_id") or "").strip(),
                text_id=(item.get("text_id") or "").strip(),
                text_hash=_sha(item.get("text_hash"), "selection text hash"),
                synthesis_text_sha256=_sha(
                    item.get("synthesis_text_sha256"), "selection synthesis text hash"
                ),
                synthesis_seed=(item.get("synthesis_seed") or "").strip(),
                normalization_operations=(item.get("normalization_operations") or "").strip(),
            )
        except ValueError as error:
            raise V4FinalMaterializationError(f"Selection row {number} is invalid.") from error
        expected_split = "test" if row.language == "ru" else "train"
        if (
            row.language not in {"ru", "kk"}
            or row.selection_rank not in range(1, 501)
            or row.source_split != expected_split
            or (row.language == "ru" and not row.synthesis_seed.isdecimal())
            or (row.language == "kk" and row.synthesis_seed)
        ):
            raise V4FinalMaterializationError("Selection language/rank/split changed.")
        for field in ("sample_id", "parent_group_id", "text_id", "text_hash", "pair_key"):
            value = cast(str, getattr(row, field))
            if not value or value in seen[field]:
                raise V4FinalMaterializationError(f"Selection repeats or omits {field}.")
            seen[field].add(value)
        selected.append(row)
    if (
        len(selected) != 1000
        or {row.language for row in selected} != {"ru", "kk"}
        or {row.selection_rank for row in selected if row.language == "ru"} != set(range(1, 501))
        or {row.selection_rank for row in selected if row.language == "kk"} != set(range(1, 501))
    ):
        raise V4FinalMaterializationError("Selection must retain exactly 500 RU plus 500 KK rows.")
    return tuple(selected)


def _require_ledger(plan: Plan, root: Path) -> None:
    ledger = load_license_ledger(
        _verify(plan.inputs["materialization_ledger"], root, "materialization ledger")
    )
    if set(ledger) != {RU_SOURCE_ID, KK_SOURCE_ID, RU_SPOOF_ID, KK_SPOOF_ID}:
        raise V4FinalMaterializationError("Materialization ledger has an unexpected source set.")
    for source_id, entry in ledger.items():
        if (
            entry.status not in APPROVED_LICENSE_STATUSES
            or entry.usage_scope != "personal_research"
            or entry.train_dev_test_use != "prohibited"
            or entry.ood_evaluation_use != "prohibited"
        ):
            raise V4FinalMaterializationError(
                f"Materialization ledger does not narrowly approve {source_id!r}."
            )


def _source_rows(
    selected: Sequence[SelectedRow],
    common_voice_archive: Path,
    fleurs_root: Path,
    data_root: Path,
    plan: Plan,
    created_at: str,
) -> tuple[tuple[ManifestRow, ...], tuple[ManifestRow, ...], dict[str, str], dict[str, str]]:
    ru = sorted(
        (row for row in selected if row.language == "ru"), key=lambda row: row.selection_rank
    )
    kk = sorted(
        (row for row in selected if row.language == "kk"), key=lambda row: row.selection_rank
    )
    records = load_common_voice_metadata_from_archive(common_voice_archive, ("test",))
    by_sample = {f"common_voice_ru_v24:{Path(record.clip_name).stem}": record for record in records}
    selected_cv: list[CommonVoiceRecord] = []
    texts: dict[str, str] = {}
    for row in ru:
        record = by_sample.get(row.sample_id)
        if record is None or record.split != "test" or record.clip_name != row.source_member:
            raise V4FinalMaterializationError(
                f"Common Voice selected source disappeared: {row.sample_id!r}."
            )
        group = f"common_voice_ru_v24:client:{record.client_id}"
        digest = hashlib.sha256(record.sentence.encode("utf-8")).hexdigest()
        if (
            group != row.parent_group_id
            or group != row.speaker_pseudo_id
            or record.sentence_id != row.text_id
            or digest != row.text_hash
            or digest != row.synthesis_text_sha256
        ):
            raise V4FinalMaterializationError(
                f"Common Voice selected metadata changed: {row.sample_id!r}."
            )
        selected_cv.append(record)
        texts[row.sample_id] = record.sentence
    ru_destination = data_root / plan.raw_roots["ru_source"]
    ru_destination.parent.mkdir(parents=True, exist_ok=True)
    extracted_cv = extract_common_voice_audio_slice(
        common_voice_archive, (record.clip_name for record in selected_cv), ru_destination
    )
    cv_assets: dict[str, ExtractedCommonVoiceAsset] = {}
    for record in selected_cv:
        asset = extracted_cv[record.clip_name]
        duration, sample_rate = inspect_extracted_common_voice_audio(asset)
        cv_assets[record.clip_name] = ExtractedCommonVoiceAsset(
            clip_name=record.clip_name,
            relative_path=asset.relative_to(data_root).as_posix(),
            sha256=sha256_file(asset),
            duration_s=duration,
            original_sr=sample_rate,
        )
    ru_rows = tuple(common_voice_manifest_rows(selected_cv, cv_assets, created_at=created_at))
    ru_rows = tuple(
        replace(
            row,
            split="test",
            source_name=RU_SOURCE_ID,
            source_license=COMMON_VOICE_RU_V24_SOURCE_LICENSE,
            rights_basis=(
                "Pinned Common Voice RU v24 exact selected final-materialization slice; "
                "personal research only; no re-hosting, speaker identification, replacement "
                "or backfill"
            ),
            code_switch="false",
        )
        for row in ru_rows
    )
    report, records_by_split = inspect_fleurs_release(fleurs_root, "kk_kz")
    if report.source_splits.get("train") != 3200:
        raise V4FinalMaterializationError(
            "FLEURS kk_kz release no longer has its pinned train split."
        )
    by_filename = {record.filename: record for record in records_by_split["train"]}
    selected_fleurs: list[FleursRecord] = []
    for row in kk:
        fleurs_record = by_filename.get(row.source_member)
        if (
            fleurs_record is None
            or f"google_fleurs_kk_v1:{fleurs_record.filename.removesuffix('.wav')}" != row.sample_id
        ):
            raise V4FinalMaterializationError(
                f"FLEURS selected source disappeared: {row.sample_id!r}."
            )
        if (
            f"google_fleurs_kk_v1:prompt:{fleurs_record.prompt_id}" != row.parent_group_id
            or row.speaker_pseudo_id != "google_fleurs_kk_v1:unknown"
            or f"google_fleurs_kk_v1:prompt:{fleurs_record.prompt_id}" != row.text_id
            or fleurs_record.text_hash != row.text_hash
        ):
            raise V4FinalMaterializationError(
                f"FLEURS selected metadata changed: {row.sample_id!r}."
            )
        normalized = normalize_kazakhtts_stage_c_text(fleurs_record.transcript, "kk")
        if normalized.normalized_sha256 != row.synthesis_text_sha256:
            raise V4FinalMaterializationError(
                f"FLEURS synthesis text binding changed: {row.sample_id!r}."
            )
        selected_fleurs.append(fleurs_record)
        texts[row.sample_id] = fleurs_record.transcript
    kk_destination = data_root / plan.raw_roots["kk_source"]
    kk_destination.parent.mkdir(parents=True, exist_ok=True)
    extracted_fleurs = extract_fleurs_audio_slice(
        fleurs_root, "kk_kz", "train", selected_fleurs, kk_destination
    )
    fleurs_assets: dict[str, FleursExtractedAsset] = {}
    for fleurs_item in selected_fleurs:
        asset = extracted_fleurs[fleurs_item.filename]
        duration, sample_rate, codec = inspect_extracted_fleurs_audio(asset)
        fleurs_assets[fleurs_item.filename] = FleursExtractedAsset(
            fleurs_item.filename,
            asset.relative_to(data_root).as_posix(),
            sha256_file(asset),
            duration,
            sample_rate,
            codec,
        )
    kk_rows = tuple(
        fleurs_manifest_rows(
            selected_fleurs, fleurs_assets, manifest_split="test", created_at=created_at
        )
    )
    kk_rows = tuple(
        replace(
            row,
            source_name=KK_SOURCE_ID,
            source_license=FLEURS_LICENSE,
            rights_basis=(
                "Pinned FLEURS kk_kz train exact selected final-materialization slice; "
                "CC-BY-4.0 attribution retained; personal research only; no replacement "
                "or backfill"
            ),
            code_switch="false",
            parent_group_id=f"google_fleurs_kk_v1:prompt:{row.text_id.rsplit(':', 1)[-1]}",
            speaker_pseudo_id="google_fleurs_kk_v1:unknown",
        )
        for row in kk_rows
    )
    source_texts = {**texts}
    synthesis_texts = {
        row.sample_id: (
            source_texts[row.sample_id]
            if row.language == "ru"
            else normalize_kazakhtts_stage_c_text(source_texts[row.sample_id], "kk").normalized
        )
        for row in selected
    }
    if {row.sample_id for row in ru_rows} != {row.sample_id for row in ru} or {
        row.sample_id for row in kk_rows
    } != {row.sample_id for row in kk}:
        raise V4FinalMaterializationError(
            "Source materialization does not cover every frozen selected row."
        )
    return ru_rows, kk_rows, source_texts, synthesis_texts


def _spoof_row(
    base: ManifestRow,
    model: ResearchTtsModel,
    source_id: str,
    relative_path: str,
    audio_sha256: str,
    duration: float,
    sample_rate: int,
    created_at: str,
    seed: str,
    synthesis_sha: str,
    device: str,
) -> ManifestRow:
    key = hashlib.sha256(
        f"{PROTOCOL_ID}:{base.sample_id}:{model.model_id}:{synthesis_sha}".encode()
    ).hexdigest()[:20]
    return ManifestRow(
        sample_id=f"{source_id}:{key}",
        relative_path=relative_path,
        sha256=audio_sha256,
        split="test",
        label="spoof",
        language=base.language,
        code_switch=base.code_switch,
        parent_group_id=f"{source_id}:fixed-profile:{model.model_id}",
        source_name=source_id,
        source_license=model.license,
        rights_basis=(
            f"One-shot local text-only derivative of frozen {base.source_name} text "
            f"{base.text_id}; no reference audio, cloning, resynthesis, replacement or backfill"
        ),
        speaker_pseudo_id=f"{source_id}:synthetic-profile:{model.model_id}",
        text_id=base.text_id,
        text_hash=base.text_hash,
        duration_s=duration,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=model.model_id,
        clone_consent_id="not_applicable:text_only_no_reference_audio",
        device=device,
        capture_route="offline_text_only_final_materialization",
        original_sr=sample_rate,
        codec="wav",
        augmentation_chain=f"synthesis_text_sha256={synthesis_sha};reference_audio=forbidden;voice_cloning=false",
        augmentation_seed=seed,
        created_at=created_at,
    )


def _synthesize_ru(
    rows: Sequence[ManifestRow],
    text_by_id: Mapping[str, str],
    selection_by_id: Mapping[str, SelectedRow],
    plan: Plan,
    root: Path,
    data_root: Path,
    created_at: str,
) -> tuple[ManifestRow, ...]:
    lock = load_research_tts_model_lock(
        _verify(plan.inputs["qwen_model_lock"], root, "Qwen model lock")
    )
    if len(lock.models) != 1:
        raise V4FinalMaterializationError("Qwen model lock must contain exactly one route.")
    model = lock.models[0]
    verify_research_tts_model_lock(
        _project_path(root, plan.model_roots["qwen"], "Qwen model root"), lock
    )
    runtime = load_qwen3_tts_customvoice(
        _project_path(root, plan.model_roots["qwen"], "Qwen model root"), model
    )
    destination = data_root / plan.raw_roots["ru_spoof"]
    journal = _project_path(root, plan.runtime_root, "runtime root") / "ru_qwen_one_shot.jsonl"
    if destination.exists() or journal.exists():
        raise V4FinalMaterializationError(
            "Qwen one-shot namespace already exists; resynthesis is forbidden."
        )
    destination.mkdir(parents=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    output: list[ManifestRow] = []
    for base in sorted(rows, key=lambda item: selection_by_id[item.sample_id].selection_rank):
        selection = selection_by_id[base.sample_id]
        prepared = runtime.prepare_text(text_by_id[base.sample_id])
        if (
            str(prepared.seed) != selection.synthesis_seed
            or hashlib.sha256(prepared.source_text.encode()).hexdigest()
            != selection.synthesis_text_sha256
        ):
            raise V4FinalMaterializationError(
                "Qwen literal text or seed diverges from metadata selection."
            )
        name = f"ru_qwen_{selection.selection_rank:03d}_{base.text_hash[:12]}.wav"
        path = destination / name
        _append_jsonl(journal, {"event": "planned", "sample_id": base.sample_id, "output": name})
        runtime.synthesize_to_file(prepared, path)
        info = sf.info(path)
        if info.samplerate != runtime.sample_rate or info.channels != 1 or info.duration <= 0:
            raise V4FinalMaterializationError("Qwen produced invalid locked WAV.")
        audio_sha = sha256_file(path)
        row = _spoof_row(
            base,
            model,
            RU_SPOOF_ID,
            path.relative_to(data_root).as_posix(),
            audio_sha,
            float(info.duration),
            int(info.samplerate),
            created_at,
            str(prepared.seed),
            selection.synthesis_text_sha256,
            "cuda:0",
        )
        output.append(replace(row, voice_id="qwen3_tts_customvoice:aiden"))
        _append_jsonl(
            journal,
            {
                "event": "generated",
                "sample_id": row.sample_id,
                "base_sample_id": base.sample_id,
                "audio_sha256": row.sha256,
            },
        )
    validate_manifest(output)
    return tuple(output)


def _synthesize_kk(
    rows: Sequence[ManifestRow],
    text_by_id: Mapping[str, str],
    selection_by_id: Mapping[str, SelectedRow],
    plan: Plan,
    root: Path,
    data_root: Path,
    created_at: str,
) -> tuple[ManifestRow, ...]:
    import math

    import numpy as np

    lock = load_research_tts_model_lock(
        _verify(plan.inputs["kazakhtts_model_lock"], root, "KazakhTTS model lock")
    )
    if len(lock.models) != 1:
        raise V4FinalMaterializationError("KazakhTTS model lock must contain exactly one route.")
    model = lock.models[0]
    runtime = load_kazakhtts_runtime(model)
    verified = verify_research_tts_model_lock(
        _project_path(root, plan.model_roots["kazakhtts"], "KazakhTTS model root"), lock
    )
    destination = data_root / plan.raw_roots["kk_spoof"]
    journal = _project_path(root, plan.runtime_root, "runtime root") / "kk_kazakhtts_one_shot.jsonl"
    if destination.exists() or journal.exists():
        raise V4FinalMaterializationError(
            "KazakhTTS one-shot namespace already exists; resynthesis is forbidden."
        )
    destination.mkdir(parents=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_kazakhtts_device("cuda")
    results: list[ManifestRow] = []
    with tempfile.TemporaryDirectory(prefix="kds-v4-final-kazakhtts-") as temporary:
        extracted = extract_verified_kazakhtts_runtime(
            verified_paths=verified[model.model_id],
            runtime=runtime,
            destination=Path(temporary) / "runtime",
        )
        text_to_speech, vocoder = load_kazakhtts_models(runtime, extracted, device)
        for base in sorted(rows, key=lambda item: selection_by_id[item.sample_id].selection_rank):
            selection = selection_by_id[base.sample_id]
            normalized = normalize_kazakhtts_stage_c_text(text_by_id[base.sample_id], "kk")
            if normalized.normalized_sha256 != selection.synthesis_text_sha256:
                raise V4FinalMaterializationError(
                    "KazakhTTS normalization diverges from metadata selection."
                )
            synthesis_input = validate_kazakhtts_text(normalized.normalized, extracted)
            if synthesis_input != normalized.normalized:
                raise V4FinalMaterializationError(
                    "KazakhTTS runtime would further rewrite frozen normalized text."
                )
            name = f"kk_kazakhtts_{selection.selection_rank:03d}_{base.text_hash[:12]}.wav"
            path = destination / name
            _append_jsonl(
                journal, {"event": "planned", "sample_id": base.sample_id, "output": name}
            )
            waveform = synthesize_kazakhtts_waveform(text_to_speech, vocoder, synthesis_input)
            if not np.isfinite(waveform).all():
                raise V4FinalMaterializationError("KazakhTTS produced non-finite waveform.")
            sf.write(path, waveform, runtime.sample_rate, subtype="PCM_16")
            info = sf.info(path)
            if (
                info.samplerate != runtime.sample_rate
                or info.channels != 1
                or not math.isfinite(info.duration)
                or info.duration <= 0
            ):
                raise V4FinalMaterializationError("KazakhTTS produced invalid locked WAV.")
            audio_sha = sha256_file(path)
            row = _spoof_row(
                base,
                model,
                KK_SPOOF_ID,
                path.relative_to(data_root).as_posix(),
                audio_sha,
                float(info.duration),
                int(info.samplerate),
                created_at,
                selection.synthesis_seed,
                selection.synthesis_text_sha256,
                str(device),
            )
            results.append(
                replace(
                    row,
                    voice_id=f"{model.model_id}:{runtime.fixed_voice_id}",
                    augmentation_chain=(
                        f"text_normalizer={KAZAKHTTS_TEXT_NORMALIZER_ID};"
                        f"synthesis_text_sha256={selection.synthesis_text_sha256};"
                        "reference_audio=forbidden;voice_cloning=false"
                    ),
                )
            )
            _append_jsonl(
                journal,
                {
                    "event": "generated",
                    "sample_id": row.sample_id,
                    "base_sample_id": base.sample_id,
                    "audio_sha256": row.sha256,
                },
            )
    validate_manifest(results)
    return tuple(results)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _decode_path(raw_sha: str, namespace: str) -> str:
    return (
        "processed/v4/xlsr_sls_model_v4_final_materialization_v1/"
        f"{namespace}/{raw_sha[:2]}/{raw_sha}.wav"
    )


def _decode(
    rows: Sequence[ManifestRow], namespace: str, data_root: Path, runtime_root: Path, workers: int
) -> dict[str, V4DecodeResult]:
    journal = runtime_root / f"{namespace}_decode_qa.jsonl"
    tasks = {
        row.sample_id: V4DecodeTask(
            row.sample_id,
            row.relative_path,
            row.sha256,
            str(resolve_asset_path(data_root, row.relative_path)),
            _decode_path(row.sha256, namespace),
            str(data_root / _decode_path(row.sha256, namespace)),
        )
        for row in rows
    }
    completed = load_v4_decode_journal(journal, tasks)
    missing = [task for identifier, task in tasks.items() if identifier not in completed]
    if not missing:
        return completed
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[V4DecodeResult], V4DecodeTask] = {
            pool.submit(run_v4_decode_task, task): task for task in missing
        }
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_v4_decode_journal(journal, result)
            completed[result.sample_id] = result
            if number % 50 == 0 or number == len(missing):
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "stage": namespace,
                            "completed": len(completed),
                            "total": len(tasks),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return completed


def _is_manifest(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return REQUIRED_FIELDS.issubset(next(csv.reader(handle), []))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4FinalMaterializationError(f"Cannot inspect manifest inventory: {path}") from error


def _history(
    plan: Plan, root: Path
) -> tuple[dict[str, tuple[str, ...]], tuple[V4AudioSignature, ...], dict[str, object]]:
    """Use the frozen complete fingerprint inventory and fail closed on unknown history."""
    refs: dict[str, set[str]] = defaultdict(set)
    files: list[dict[str, object]] = []
    for path in sorted((root / "data/manifests").rglob("*.csv")):
        if not _is_manifest(path):
            continue
        rows = load_manifest(path)
        validate_manifest(rows)
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "sha256": sha256_file(path), "rows": len(rows)})
        for row in rows:
            refs[row.sha256].add(f"{relative}:{row.sample_id}")
    known: set[str] = set()
    signatures: dict[str, V4AudioSignature] = {}
    canonical_by_raw: dict[str, set[str]] = defaultdict(set)

    def add(raw_sha: str, canonical: str, fingerprint: str, speech: float) -> None:
        signature = V4AudioSignature(f"history:{canonical}", canonical, fingerprint, speech)
        known.update((raw_sha, canonical))
        canonical_by_raw[raw_sha].add(canonical)
        for key in (raw_sha, canonical):
            prior = signatures.get(key)
            if prior is not None and prior != signature:
                raise V4FinalMaterializationError(
                    "Historical audio fingerprint evidence conflicts."
                )
            signatures[key] = signature

    inventory = _verify(
        plan.inputs["historical_fingerprint_inventory"], root, "historical fingerprint inventory"
    )
    with inventory.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "manifest_audio_sha256",
            "canonical_audio_sha256",
            "speech_seconds",
            "audio_fingerprint_v1",
            "fingerprint_status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise V4FinalMaterializationError("Historical fingerprint inventory schema is invalid.")
        for item in reader:
            raw_sha = _sha(item["manifest_audio_sha256"], "historical raw SHA")
            known.add(raw_sha)
            if item["fingerprint_status"] == "fingerprinted":
                add(
                    raw_sha,
                    _sha(item["canonical_audio_sha256"], "historical canonical SHA"),
                    item["audio_fingerprint_v1"],
                    float(item["speech_seconds"]),
                )
    for name in ("source_decode_inventory", "kk_spoof_decode_inventory"):
        path = _verify(plan.inputs[name], root, name)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for item in csv.DictReader(handle):
                raw_sha = _sha(item["raw_audio_sha256"], f"{name} raw SHA")
                known.add(raw_sha)
                if item.get("decoded_audio_sha256"):
                    add(
                        raw_sha,
                        _sha(item["decoded_audio_sha256"], f"{name} decoded SHA"),
                        item["audio_fingerprint_v1"],
                        float(item["speech_seconds"]),
                    )
    for name in ("dev_source_decode_journal", "dev_spoof_decode_journal"):
        path = _verify(plan.inputs[name], root, name)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                item = cast(dict[str, object], json.loads(line))
                raw_sha = _sha(item["raw_sha256"], f"{name} raw SHA")
                known.add(raw_sha)
                decoded = item.get("decoded_audio_sha256")
                if isinstance(decoded, str) and decoded:
                    fingerprint = item.get("audio_fingerprint_v1")
                    speech = item.get("speech_seconds")
                    if not isinstance(fingerprint, str) or not isinstance(speech, (int, float)):
                        raise ValueError
                    add(raw_sha, _sha(decoded, f"{name} decoded SHA"), fingerprint, float(speech))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise V4FinalMaterializationError(f"{name} line {number} is invalid.") from error
    missing = set(refs).difference(known)
    if missing:
        raise V4FinalMaterializationError(
            "Current history has "
            f"{len(missing)} audio hashes without frozen exact/fingerprint coverage."
        )
    unique = {signature.identity: signature for signature in signatures.values()}
    exact_refs: dict[str, set[str]] = defaultdict(set)
    for raw_sha, owners in refs.items():
        exact_refs[raw_sha].update(owners)
        for canonical_sha in canonical_by_raw.get(raw_sha, ()):
            exact_refs[canonical_sha].update(owners)
    return (
        {item: tuple(sorted(value)) for item, value in exact_refs.items()},
        tuple(unique.values()),
        {
            "manifest_files": files,
            "manifest_rows": sum(cast(int, item["rows"]) for item in files),
            "unique_audio_hashes": len(refs),
            "near_fingerprint_covered_hashes": len(set(refs).intersection(signatures)),
            "exact_only_hashes": len(set(refs).difference(signatures)),
        },
    )


def _decisions(
    rows: Sequence[ManifestRow],
    results: Mapping[str, V4DecodeResult],
    ranks: Mapping[str, int],
    historical_exact: Mapping[str, Sequence[str]],
    historical_signatures: Sequence[V4AudioSignature],
) -> tuple[V4DecodedDecision, ...]:
    candidates = [
        V4DecodedCandidate(ranks[row.sample_id], row.language, row.label, results[row.sample_id])
        for row in rows
    ]
    decisions = list(
        decide_v4_decoded_audio_eligibility(candidates, historical_exact, historical_signatures)
    )
    result: list[V4DecodedDecision] = []
    for decision in decisions:
        if historical_exact.get(decision.candidate.result.raw_sha256):
            result.append(
                replace(
                    decision,
                    eligibility_status="rejected",
                    rejection_reason="historical_exact_raw_audio",
                )
            )
        else:
            result.append(decision)
    return tuple(result)


def _ready(
    raw: Mapping[str, ManifestRow], decisions: Sequence[V4DecodedDecision], created_at: str
) -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    for decision in decisions:
        if decision.eligibility_status != "eligible":
            continue
        result = decision.candidate.result
        rows.append(
            replace(
                raw[result.sample_id],
                relative_path=result.decoded_relative_path,
                sha256=result.decoded_audio_sha256,
                duration_s=result.duration_s,
                original_sr=16000,
                codec="wav",
                created_at=created_at,
            )
        )
    validate_manifest(rows)
    return tuple(rows)


def _inventory(
    decisions: Sequence[V4DecodedDecision],
    pair_by_sample: Mapping[str, str],
    historical_exact: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    return [
        {
            "selection_rank": decision.candidate.selection_rank,
            "sample_id": result.sample_id,
            "pair_key": pair_by_sample[result.sample_id],
            "language": decision.candidate.language,
            "label": decision.candidate.label,
            "raw_relative_path": result.raw_relative_path,
            "raw_audio_sha256": result.raw_sha256,
            "decoded_relative_path": result.decoded_relative_path,
            "decoded_audio_sha256": result.decoded_audio_sha256,
            "duration_s": result.duration_s,
            "peak": result.peak,
            "rms_dbfs": result.rms_dbfs,
            "clipped_fraction": result.clipped_fraction,
            "dc_offset": result.dc_offset,
            "speech_seconds": result.speech_seconds,
            "speech_segment_count": result.speech_segment_count,
            "audio_fingerprint_v1": result.audio_fingerprint_v1,
            "preparation_status": result.preparation_status,
            "eligibility_status": decision.eligibility_status,
            "rejection_reason": decision.rejection_reason,
            "historical_raw_exact_match_count": len(historical_exact.get(result.raw_sha256, ())),
            "historical_decoded_exact_match_count": len(decision.historical_exact_matches),
            "historical_near_match_count": len(decision.historical_near_matches),
            "within_pool_near_match_count": len(decision.within_pool_near_matches),
        }
        for decision in decisions
        for result in (decision.candidate.result,)
    ]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _review_packet(
    rows: Sequence[ManifestRow],
    receipt_sha: str,
    source_text: Mapping[str, str],
    synthesis_text: Mapping[str, str],
    pair_by_sample: Mapping[str, str],
    data_root: Path,
) -> tuple[ReviewPacketRow, ...]:
    packet: list[ReviewPacketRow] = []
    bases_by_text = {row.text_id: row for row in rows if row.label == "bonafide"}
    for row in sorted(rows, key=lambda item: (item.language, item.text_id, item.label)):
        base = row if row.label == "bonafide" else bases_by_text.get(row.text_id)
        if base is None:
            raise V4FinalMaterializationError("Ready spoof row has no paired ready source row.")
        packet.append(
            ReviewPacketRow(
                PROTOCOL_ID,
                receipt_sha,
                row.sample_id,
                pair_by_sample[row.sample_id],
                row.language,
                row.label,
                row.text_id,
                row.text_hash,
                str(resolve_asset_path(data_root, row.relative_path)),
                row.sha256,
                source_text[base.sample_id],
                synthesis_text[base.sample_id] if row.label == "spoof" else "",
                hashlib.sha256(synthesis_text[base.sample_id].encode()).hexdigest()
                if row.label == "spoof"
                else "",
            )
        )
    return tuple(packet)


def _stage_outputs(root: Path, outputs: Mapping[str, str]) -> Path:
    targets = [_project_path(root, value, f"output.{name}") for name, value in outputs.items()]
    if any(path.exists() or not path.parent.is_dir() for path in targets):
        raise V4FinalMaterializationError(
            "Every materialization output must be new and have an existing parent."
        )
    return Path(tempfile.mkdtemp(prefix=".kds-v4-final-materialization-", dir=targets[0].parent))


def _verify_common_voice_archive(common_voice_archive: Path) -> None:
    if (
        not common_voice_archive.is_file()
        or common_voice_archive.name != "cv-corpus-24.0-2025-12-05-ru.tar.gz"
        or common_voice_archive.stat().st_size != 7_008_716_262
        or sha256_file(common_voice_archive)
        != "9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea"
    ):
        raise V4FinalMaterializationError(
            "Common Voice archive identity does not match the frozen contract."
        )


def _validate_selected_source_metadata(
    selected: Sequence[SelectedRow], common_voice_archive: Path, fleurs_root: Path
) -> None:
    """Prove all frozen source identities before any destination is created."""

    records = load_common_voice_metadata_from_archive(common_voice_archive, ("test",))
    by_sample = {f"common_voice_ru_v24:{Path(record.clip_name).stem}": record for record in records}
    for row in (item for item in selected if item.language == "ru"):
        record = by_sample.get(row.sample_id)
        if record is None or record.split != "test" or record.clip_name != row.source_member:
            raise V4FinalMaterializationError(
                f"Common Voice selected source disappeared: {row.sample_id!r}."
            )
        digest = hashlib.sha256(record.sentence.encode("utf-8")).hexdigest()
        if (
            f"common_voice_ru_v24:client:{record.client_id}" != row.parent_group_id
            or row.parent_group_id != row.speaker_pseudo_id
            or record.sentence_id != row.text_id
            or digest != row.text_hash
            or digest != row.synthesis_text_sha256
        ):
            raise V4FinalMaterializationError(
                f"Common Voice selected metadata changed: {row.sample_id!r}."
            )
    report, records_by_split = inspect_fleurs_release(fleurs_root, "kk_kz")
    if report.source_splits.get("train") != 3200:
        raise V4FinalMaterializationError(
            "FLEURS kk_kz release no longer has its pinned train split."
        )
    by_filename = {record.filename: record for record in records_by_split["train"]}
    for row in (item for item in selected if item.language == "kk"):
        fleurs_record = by_filename.get(row.source_member)
        if (
            fleurs_record is None
            or f"google_fleurs_kk_v1:{fleurs_record.filename.removesuffix('.wav')}" != row.sample_id
            or f"google_fleurs_kk_v1:prompt:{fleurs_record.prompt_id}" != row.parent_group_id
            or row.speaker_pseudo_id != "google_fleurs_kk_v1:unknown"
            or f"google_fleurs_kk_v1:prompt:{fleurs_record.prompt_id}" != row.text_id
            or fleurs_record.text_hash != row.text_hash
            or normalize_kazakhtts_stage_c_text(fleurs_record.transcript, "kk").normalized_sha256
            != row.synthesis_text_sha256
        ):
            raise V4FinalMaterializationError(
                f"FLEURS selected metadata changed: {row.sample_id!r}."
            )


def _preflight_tts_routes(plan: Plan, root: Path) -> None:
    """Verify/load each local TTS route without synthesizing an audio asset."""

    qwen_lock = load_research_tts_model_lock(
        _verify(plan.inputs["qwen_model_lock"], root, "Qwen model lock")
    )
    if len(qwen_lock.models) != 1:
        raise V4FinalMaterializationError("Qwen model lock must contain exactly one route.")
    qwen_root = _project_path(root, plan.model_roots["qwen"], "Qwen model root")
    verify_research_tts_model_lock(qwen_root, qwen_lock)
    load_qwen3_tts_customvoice(qwen_root, qwen_lock.models[0])

    kazakhtts_lock = load_research_tts_model_lock(
        _verify(plan.inputs["kazakhtts_model_lock"], root, "KazakhTTS model lock")
    )
    if len(kazakhtts_lock.models) != 1:
        raise V4FinalMaterializationError("KazakhTTS model lock must contain exactly one route.")
    kazakhtts_root = _project_path(root, plan.model_roots["kazakhtts"], "KazakhTTS model root")
    verified = verify_research_tts_model_lock(kazakhtts_root, kazakhtts_lock)
    runtime = load_kazakhtts_runtime(kazakhtts_lock.models[0])
    device = resolve_kazakhtts_device("cuda")
    with tempfile.TemporaryDirectory(prefix="kds-v4-final-kazakhtts-preflight-") as temporary:
        extracted = extract_verified_kazakhtts_runtime(
            verified_paths=verified[kazakhtts_lock.models[0].model_id],
            runtime=runtime,
            destination=Path(temporary) / "runtime",
        )
        load_kazakhtts_models(runtime, extracted, device)


def preflight_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
    created_at: str,
) -> Plan:
    """Check every non-audio dependency before the one-shot pass is permitted."""

    _timestamp(created_at, "created_at")
    root = project_root.resolve(strict=True)
    plan = load_plan(plan_path, root)
    _require_ledger(plan, root)
    relative_data = _project_path(root, "data", "data root")
    if data_root.resolve(strict=True) != relative_data:
        raise V4FinalMaterializationError("data_root must be the project data directory.")
    _verify_common_voice_archive(common_voice_archive)
    selected = _load_selection(plan, root)
    _validate_selected_source_metadata(selected, common_voice_archive, fleurs_release_root)
    _history(plan, root)
    runtime = _project_path(root, plan.runtime_root, "runtime root")
    if runtime.exists():
        raise V4FinalMaterializationError("Final materialization runtime namespace already exists.")
    for value in plan.raw_roots.values():
        if (data_root / value).exists():
            raise V4FinalMaterializationError("Raw materialization destination is not new.")
    for name, relative in plan.outputs.items():
        target = _project_path(root, relative, f"output.{name}")
        if target.exists() or not target.parent.is_dir():
            raise V4FinalMaterializationError(
                "Every materialization and pair-lock output must be new with an existing parent."
            )
    _preflight_tts_routes(plan, root)
    return plan


def run_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
    workers: int,
    created_at: str,
) -> Plan:
    """Execute the only extraction/synthesis pass.  It never writes a pair lock."""

    if workers <= 0:
        raise V4FinalMaterializationError("workers must be positive.")
    root = project_root.resolve(strict=True)
    plan = preflight_materialization(
        plan_path=plan_path,
        project_root=root,
        data_root=data_root,
        common_voice_archive=common_voice_archive,
        fleurs_release_root=fleurs_release_root,
        created_at=created_at,
    )
    selected = _load_selection(plan, root)
    _project_path(root, plan.runtime_root, "runtime root").mkdir(parents=True)
    ru_source, kk_source, source_text, synthesis_text = _source_rows(
        selected, common_voice_archive, fleurs_release_root, data_root, plan, created_at
    )
    selection_by_id = {row.sample_id: row for row in selected}
    ru_spoof = _synthesize_ru(
        ru_source, source_text, selection_by_id, plan, root, data_root, created_at
    )
    kk_spoof = _synthesize_kk(
        kk_source, source_text, selection_by_id, plan, root, data_root, created_at
    )
    groups = {
        "ru_source": ru_source,
        "kk_source": kk_source,
        "ru_spoof": ru_spoof,
        "kk_spoof": kk_spoof,
    }
    ledger = load_license_ledger(
        _verify(plan.inputs["materialization_ledger"], root, "materialization ledger")
    )
    for rows in groups.values():
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
        require_valid_assets(rows, data_root)
    runtime = _project_path(root, plan.runtime_root, "runtime root")
    decoded = {
        name: _decode(rows, name, data_root, runtime, workers) for name, rows in groups.items()
    }
    historical_exact, historical_signatures, history = _history(plan, root)
    ranks = {row.sample_id: row.selection_rank for row in selected}
    pair_by_sample: dict[str, str] = {row.sample_id: row.pair_key for row in selected}
    for source_rows, spoof_rows in ((ru_source, ru_spoof), (kk_source, kk_spoof)):
        by_text = {row.text_id: row for row in source_rows}
        for spoof in spoof_rows:
            source = by_text.get(spoof.text_id)
            if source is None or source.text_hash != spoof.text_hash:
                raise V4FinalMaterializationError("One-shot spoof route broke source text pairing.")
            pair_by_sample[spoof.sample_id] = pair_by_sample[source.sample_id]
            ranks[spoof.sample_id] = ranks[source.sample_id]
    decisions = {
        name: _decisions(rows, decoded[name], ranks, historical_exact, historical_signatures)
        for name, rows in groups.items()
    }
    ready = {
        name: _ready({row.sample_id: row for row in rows}, decisions[name], created_at)
        for name, rows in groups.items()
    }
    all_ready = tuple(
        item for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof") for item in ready[name]
    )
    for rows in ready.values():
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
        require_valid_assets(rows, data_root)
    stage = _stage_outputs(
        root,
        {
            name: value
            for name, value in plan.outputs.items()
            if name not in {"pair_lock_manifest", "pair_lock_receipt"}
        },
    )
    try:
        staged = {name: stage / Path(value).name for name, value in plan.outputs.items()}
        for name, rows in groups.items():
            write_manifest(staged[f"{name}_raw_manifest"], rows)
        for name, rows in ready.items():
            write_manifest(staged[f"{name}_ready_manifest"], rows)
        all_decisions = tuple(
            item
            for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof")
            for item in decisions[name]
        )
        _write_csv(
            staged["audio_inventory"],
            INVENTORY_FIELDS,
            _inventory(all_decisions, pair_by_sample, historical_exact),
        )
        receipt = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "materialized_review_required_pair_lock_pending",
            "created_at": created_at,
            "plan": {"path": plan.path, "sha256": plan.sha256},
            "inputs": {
                name: {"path": item.path, "sha256": item.sha256, "rows": item.rows}
                for name, item in sorted(plan.inputs.items())
            },
            "counts": {
                "selected_source_rows": {"ru": 500, "kk": 500},
                "one_shot_synthetic_rows": {"ru": len(ru_spoof), "kk": len(kk_spoof)},
                "raw_assets": sum(len(value) for value in groups.values()),
                "eligible_assets": len(all_ready),
                "eligible_by_cell": {
                    f"{language}/{label}": sum(
                        row.language == language and row.label == label for row in all_ready
                    )
                    for language in ("ru", "kk")
                    for label in ("bonafide", "spoof")
                },
            },
            "history": history,
            "outputs": {
                name: {
                    "path": plan.outputs[name],
                    "sha256": sha256_file(staged[name]),
                    "rows": _rows(staged[name]) if staged[name].suffix == ".csv" else None,
                }
                for name in (
                    "ru_source_raw_manifest",
                    "kk_source_raw_manifest",
                    "ru_spoof_raw_manifest",
                    "kk_spoof_raw_manifest",
                    "ru_source_ready_manifest",
                    "kk_source_ready_manifest",
                    "ru_spoof_ready_manifest",
                    "kk_spoof_ready_manifest",
                    "audio_inventory",
                )
            },
            "claims": {
                "raw_audio_extraction_performed": True,
                "synthetic_audio_generated": True,
                "technical_decode_qa_vad_performed": True,
                "full_history_audio_isolation_performed": True,
                "acoustic_review_performed": False,
                "pair_lock_performed": False,
                "detector_checkpoint_loaded": False,
                "calibration_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "final_inference_performed": False,
                "replacement_or_backfill": False,
                "resynthesis": False,
            },
        }
        staged["materialization_receipt"].write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_sha = sha256_file(staged["materialization_receipt"])
        packet = _review_packet(
            all_ready, receipt_sha, source_text, synthesis_text, pair_by_sample, data_root
        )
        _write_csv(staged["review_packet"], PACKET_FIELDS, [asdict(row) for row in packet])
        for name, reviewer in (
            ("reviewer_a_template", "reviewer_A_REPLACE_ME"),
            ("reviewer_b_template", "reviewer_B_REPLACE_ME"),
        ):
            _write_csv(
                staged[name],
                REVIEW_FIELDS,
                [
                    {
                        "protocol_id": PROTOCOL_ID,
                        "packet_sha256": sha256_file(staged["review_packet"]),
                        "sample_id": row.sample_id,
                        "audio_sha256": row.audio_sha256,
                        "reviewer_pseudo_id": reviewer,
                        "review_status": "inconclusive",
                        "speech_intelligible": "unknown",
                        "lexical_content_preserved": "unknown",
                        "language_preserved": "unknown",
                        "severe_artifacts": "unknown",
                        "notes": "",
                    }
                    for row in packet
                ],
            )
        to_publish = (
            "ru_source_raw_manifest",
            "kk_source_raw_manifest",
            "ru_spoof_raw_manifest",
            "kk_spoof_raw_manifest",
            "ru_source_ready_manifest",
            "kk_source_ready_manifest",
            "ru_spoof_ready_manifest",
            "kk_spoof_ready_manifest",
            "audio_inventory",
            "review_packet",
            "reviewer_a_template",
            "reviewer_b_template",
            "materialization_receipt",
        )
        if any(
            _project_path(root, plan.outputs[name], f"output.{name}").exists()
            for name in to_publish
        ):
            raise V4FinalMaterializationError("An output appeared during staging.")
        for name in to_publish:
            staged[name].replace(_project_path(root, plan.outputs[name], f"output.{name}"))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return plan


def _read_packet(path: Path, materialization_sha: str) -> tuple[ReviewPacketRow, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PACKET_FIELDS:
                raise V4FinalMaterializationError("Review packet columns are invalid.")
            rows = tuple(ReviewPacketRow(**item) for item in reader)
    except OSError as error:
        raise V4FinalMaterializationError("Cannot read review packet.") from error
    if (
        not rows
        or len({row.sample_id for row in rows}) != len(rows)
        or any(
            row.protocol_id != PROTOCOL_ID
            or row.materialization_receipt_sha256 != materialization_sha
            or not Path(row.audio_path).is_file()
            or sha256_file(Path(row.audio_path)) != row.audio_sha256
            for row in rows
        )
    ):
        raise V4FinalMaterializationError("Review packet does not bind exact available audio.")
    return rows


def _read_review(path: Path) -> tuple[ReviewRow, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
                raise V4FinalMaterializationError("Review form columns are invalid.")
            return tuple(ReviewRow(**item) for item in reader)
    except OSError as error:
        raise V4FinalMaterializationError("Cannot read review form.") from error


def _bound_review_path(
    root: Path, plan: Plan, template_name: str, supplied: Path, label: str
) -> Path:
    expected = _project_path(root, plan.outputs[template_name], label)
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        actual = candidate.resolve(strict=True)
    except OSError as error:
        raise V4FinalMaterializationError(f"Cannot resolve {label}.") from error
    if actual != expected:
        raise V4FinalMaterializationError(
            f"{label} must be the exact versioned {template_name} file."
        )
    return actual


def _validate_materialization_receipt(
    receipt: Mapping[str, object], plan: Plan, root: Path
) -> None:
    claims = _mapping(receipt.get("claims"), "materialization receipt claims")
    receipt_plan = _mapping(receipt.get("plan"), "materialization receipt plan")
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "materialized_review_required_pair_lock_pending"
        or receipt_plan.get("path") != plan.path
        or receipt_plan.get("sha256") != plan.sha256
        or claims.get("technical_decode_qa_vad_performed") is not True
        or claims.get("full_history_audio_isolation_performed") is not True
        or claims.get("acoustic_review_performed") is not False
        or claims.get("pair_lock_performed") is not False
        or claims.get("detector_checkpoint_loaded") is not False
        or claims.get("final_inference_performed") is not False
        or claims.get("detector_inference_performed") is not False
    ):
        raise V4FinalMaterializationError(
            "Materialization receipt claims do not permit pair locking."
        )
    outputs = _mapping(receipt.get("outputs"), "materialization receipt outputs")
    expected_outputs = {
        "ru_source_raw_manifest",
        "kk_source_raw_manifest",
        "ru_spoof_raw_manifest",
        "kk_spoof_raw_manifest",
        "ru_source_ready_manifest",
        "kk_source_ready_manifest",
        "ru_spoof_ready_manifest",
        "kk_spoof_ready_manifest",
        "audio_inventory",
    }
    if set(outputs) != expected_outputs:
        raise V4FinalMaterializationError("Materialization receipt output set is incomplete.")
    for name, raw_output in outputs.items():
        binding = _mapping(raw_output, f"materialization receipt outputs.{name}")
        target = _project_path(root, plan.outputs[name], f"output.{name}")
        if (
            set(binding) != {"path", "sha256", "rows"}
            or binding.get("path") != plan.outputs[name]
            or binding.get("sha256") != sha256_file(target)
            or binding.get("rows") != _rows(target)
        ):
            raise V4FinalMaterializationError(f"Materialization receipt no longer binds {name}.")


def _reviewer_id(
    rows: Sequence[ReviewRow], packet_by_id: Mapping[str, ReviewPacketRow], packet_sha: str
) -> str:
    if len(rows) != len(packet_by_id) or len({row.sample_id for row in rows}) != len(rows):
        raise V4FinalMaterializationError("Each review form must contain every packet asset once.")
    identifiers = {row.reviewer_pseudo_id for row in rows}
    if len(identifiers) != 1:
        raise V4FinalMaterializationError("A review form must have exactly one reviewer identity.")
    reviewer = next(iter(identifiers))
    if not reviewer or "REPLACE_ME" in reviewer:
        raise V4FinalMaterializationError("Review form has no usable reviewer identity.")
    for row in rows:
        item = packet_by_id.get(row.sample_id)
        if (
            item is None
            or row.protocol_id != PROTOCOL_ID
            or row.packet_sha256 != packet_sha
            or row.audio_sha256 != item.audio_sha256
        ):
            raise V4FinalMaterializationError("Review row is not bound to the exact review packet.")
    return reviewer


def finalize_pair_lock(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    reviewer_a: Path,
    reviewer_b: Path,
    created_at: str,
) -> Plan:
    """Create immutable exact pairs after two complete independent reviews only."""

    _timestamp(created_at, "created_at")
    root = project_root.resolve(strict=True)
    plan = load_plan(plan_path, root)
    _require_ledger(plan, root)
    if data_root.resolve(strict=True) != _project_path(root, "data", "data root"):
        raise V4FinalMaterializationError("data_root must be the project data directory.")
    materialization = _project_path(
        root, plan.outputs["materialization_receipt"], "materialization receipt"
    )
    receipt = _object(materialization, "materialization receipt")
    _validate_materialization_receipt(receipt, plan, root)
    materialization_sha = sha256_file(materialization)
    manifests = {
        name: load_manifest(
            _project_path(root, plan.outputs[f"{name}_ready_manifest"], f"{name} ready manifest")
        )
        for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof")
    }
    for rows in manifests.values():
        validate_manifest(rows)
    all_ready = tuple(row for rows in manifests.values() for row in rows)
    if len({row.sample_id for row in all_ready}) != len(all_ready):
        raise V4FinalMaterializationError("Ready manifests repeat a sample ID.")
    packet_path = _project_path(root, plan.outputs["review_packet"], "review packet")
    packet = _read_packet(packet_path, materialization_sha)
    packet_sha = sha256_file(packet_path)
    packet_by_id = {row.sample_id: row for row in packet}
    manifest_by_id = {row.sample_id: row for row in all_ready}
    if set(packet_by_id) != set(manifest_by_id):
        raise V4FinalMaterializationError("Review packet must cover every and only ready asset.")
    selected = _load_selection(plan, root)
    selected_by_text = {row.text_id: row for row in selected}
    for sample_id, item in packet_by_id.items():
        manifest = manifest_by_id[sample_id]
        selected_row = selected_by_text.get(manifest.text_id)
        if (
            selected_row is None
            or item.pair_key != selected_row.pair_key
            or item.language != manifest.language
            or item.label != manifest.label
            or item.text_id != manifest.text_id
            or item.text_hash != manifest.text_hash
            or item.audio_sha256 != manifest.sha256
            or hashlib.sha256(item.source_text.encode("utf-8")).hexdigest() != manifest.text_hash
            or (
                manifest.label == "bonafide" and (item.synthesis_text or item.synthesis_text_sha256)
            )
            or (
                manifest.label == "spoof"
                and item.synthesis_text_sha256 != selected_row.synthesis_text_sha256
            )
            or (
                manifest.label == "spoof"
                and hashlib.sha256(item.synthesis_text.encode("utf-8")).hexdigest()
                != selected_row.synthesis_text_sha256
            )
        ):
            raise V4FinalMaterializationError(
                "Review packet metadata no longer binds a ready asset."
            )
    reviewer_a_path = _bound_review_path(
        root, plan, "reviewer_a_template", reviewer_a, "reviewer A form"
    )
    reviewer_b_path = _bound_review_path(
        root, plan, "reviewer_b_template", reviewer_b, "reviewer B form"
    )
    reviews_a = _read_review(reviewer_a_path)
    reviews_b = _read_review(reviewer_b_path)
    reviewer_a_id = _reviewer_id(reviews_a, packet_by_id, packet_sha)
    reviewer_b_id = _reviewer_id(reviews_b, packet_by_id, packet_sha)
    if reviewer_a_id == reviewer_b_id:
        raise V4FinalMaterializationError("The two review forms must name distinct reviewers.")
    reviews = (*reviews_a, *reviews_b)
    by_sample: dict[str, list[ReviewRow]] = defaultdict(list)
    for row in reviews:
        by_sample[row.sample_id].append(row)
    if set(by_sample) != set(packet_by_id):
        raise V4FinalMaterializationError(
            "Two distinct reviewers must submit one review for every packet asset."
        )
    approved: set[str] = set()
    for sample_id, review_rows in by_sample.items():
        if len(review_rows) != 2 or len({row.reviewer_pseudo_id for row in review_rows}) != 2:
            raise V4FinalMaterializationError("Each packet asset needs two distinct reviews.")
        if all(
            row.review_status == "pass"
            and row.speech_intelligible == "yes"
            and row.lexical_content_preserved == "yes"
            and row.language_preserved == "yes"
            and row.severe_artifacts == "no"
            for row in review_rows
        ):
            approved.add(sample_id)
    by_text = {
        (row.language, row.text_id, row.label): row for rows in manifests.values() for row in rows
    }
    if len(by_text) != len(all_ready):
        raise V4FinalMaterializationError("Ready manifests repeat a language/text/label route.")
    pairs: list[ManifestRow] = []
    for language in ("ru", "kk"):
        for source in sorted(
            (row for row in manifests[f"{language}_source"]), key=lambda row: row.text_id
        ):
            spoof = by_text.get((language, source.text_id, "spoof"))
            if (
                spoof is not None
                and source.sample_id in approved
                and spoof.sample_id in approved
                and source.text_hash == spoof.text_hash
            ):
                pairs.extend((source, spoof))
    if not pairs:
        raise V4FinalMaterializationError(
            "No complete two-reviewed isolated pairs remain; refusing an empty lock."
        )
    validate_manifest(pairs)
    ledger = load_license_ledger(
        _verify(plan.inputs["materialization_ledger"], root, "materialization ledger")
    )
    validate_manifest_licenses(pairs, ledger)
    require_valid_assets(pairs, data_root)
    lock_path = _project_path(root, plan.outputs["pair_lock_manifest"], "pair lock manifest")
    receipt_path = _project_path(root, plan.outputs["pair_lock_receipt"], "pair lock receipt")
    if (
        lock_path.exists()
        or receipt_path.exists()
        or not lock_path.parent.is_dir()
        or not receipt_path.parent.is_dir()
    ):
        raise V4FinalMaterializationError("Pair lock outputs must be new with existing parents.")
    with tempfile.TemporaryDirectory(
        prefix=".kds-v4-final-pair-lock-", dir=lock_path.parent
    ) as stage_name:
        stage = Path(stage_name)
        staged_lock = stage / lock_path.name
        staged_receipt = stage / receipt_path.name
        write_manifest(staged_lock, pairs)
        report = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "pair_lock_complete_final_inference_still_forbidden",
            "created_at": created_at,
            "plan": {"path": plan.path, "sha256": plan.sha256},
            "inputs": {
                "materialization_receipt": {
                    "path": plan.outputs["materialization_receipt"],
                    "sha256": sha256_file(materialization),
                },
                "review_packet": {"path": plan.outputs["review_packet"], "sha256": packet_sha},
                "reviewer_a": {
                    "path": plan.outputs["reviewer_a_template"],
                    "sha256": sha256_file(reviewer_a_path),
                    "reviewer_pseudo_id": reviewer_a_id,
                },
                "reviewer_b": {
                    "path": plan.outputs["reviewer_b_template"],
                    "sha256": sha256_file(reviewer_b_path),
                    "reviewer_pseudo_id": reviewer_b_id,
                },
            },
            "counts": {
                "review_packet_assets": len(packet),
                "two_reviewer_passed_assets": len(approved),
                "locked_pairs": len(pairs) // 2,
                "locked_assets": len(pairs),
                "locked_pairs_by_language": {
                    language: sum(
                        row.language == language and row.label == "bonafide" for row in pairs
                    )
                    for language in ("ru", "kk")
                },
            },
            "output": {
                "path": plan.outputs["pair_lock_manifest"],
                "sha256": sha256_file(staged_lock),
                "rows": len(pairs),
            },
            "claims": {
                "independent_acoustic_language_review_performed": True,
                "pair_lock_performed": True,
                "detector_checkpoint_loaded": False,
                "calibration_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "final_inference_performed": False,
                "final_inference_requires_separate_contract": True,
                "replacement_or_backfill": False,
            },
        }
        staged_receipt.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_lock.replace(lock_path)
        staged_receipt.replace(receipt_path)
    return plan
