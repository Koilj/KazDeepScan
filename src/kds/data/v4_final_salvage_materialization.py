"""One-shot salvage materialization after the exhausted recovery attempt.

This contract is deliberately narrower than the recovery contract.  It reuses
only the byte-verified partial recovery outputs, permanently rejects the two
Kazakh prompts which the locked model cannot tokenize, and synthesizes the 227
remaining prevalidated Kazakh rows exactly once.  No detector, calibration, or
final inference route is present here.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data import v4_final_materialization as base
from kds.data.assets import require_valid_assets, sha256_file
from kds.data.common_voice import (
    CommonVoiceRecord,
    ExtractedCommonVoiceAsset,
    common_voice_manifest_rows,
    inspect_extracted_common_voice_audio,
    load_common_voice_metadata_from_archive,
)
from kds.data.fleurs import (
    FleursExtractedAsset,
    FleursRecord,
    fleurs_manifest_rows,
    inspect_extracted_fleurs_audio,
    inspect_fleurs_release,
)
from kds.data.kazakhtts import (
    extract_verified_kazakhtts_runtime,
    load_kazakhtts_runtime,
    validate_kazakhtts_text,
)
from kds.data.kazakhtts_inference import load_kazakhtts_models, resolve_kazakhtts_device
from kds.data.kazakhtts_text import (
    KAZAKHTTS_TEXT_NORMALIZER_ID,
    normalize_kazakhtts_stage_c_text,
)
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestRow, validate_manifest, write_manifest
from kds.data.research_tts import load_research_tts_model_lock, verify_research_tts_model_lock
from kds.data.v4_final_inputs import V4_FINAL_SELECTION_FIELDS
from kds.data.v4_final_recovery_materialization import _decode, _stage_outputs

PROTOCOL_ID = "xlsr-sls-model-v4-final-salvage-materialization-v1"
SALVAGE_AUTHORIZATION_PROTOCOL_ID = "xlsr-sls-model-v4-final-salvage-authorization-v1"
RU_SOURCE_ID = "common_voice_ru_v24_v4_final_salvage"
KK_SOURCE_ID = "google_fleurs_kk_v1_v4_final_salvage"
RU_SPOOF_ID = "qwen3_tts_customvoice_aiden_v4_final_salvage"
KK_SPOOF_ID = "issai_kazakhtts2_male2_tacotron2_pwg_v4_final_salvage"

PARTIAL_RAW_ROOTS = {
    "ru_source": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/ru_source",
    "kk_source": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/kk_source",
    "ru_spoof": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/ru_qwen",
    "kk_spoof": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/kk_kazakhtts",
}
RAW_ROOTS = {
    "kk_spoof": "raw/v4/xlsr_sls_model_v4_final_salvage_materialization_v1/kk_kazakhtts_remaining",
}
PROCESSED_ROOT = "processed/v4/xlsr_sls_model_v4_final_salvage_materialization_v1"
RUNTIME_ROOT = "artifacts/v4/xlsr_sls_model_v4_final_salvage_materialization_v1"
MODEL_ROOTS = {
    "qwen": "models/research/voxforge_ru_mdc_qwen3_tts_customvoice_aiden",
    "kazakhtts": "models/research/kazakhtts_tacotron2_pwg_v1",
}
OUTPUTS = {
    "ru_source_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_ru_source_raw_v1.csv",
    "kk_source_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_kk_source_raw_v1.csv",
    "ru_spoof_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_ru_qwen_raw_v1.csv",
    "kk_spoof_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_kk_kazakhtts_raw_v1.csv",
    "ru_source_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_ru_source_ready_v1.csv",
    "kk_source_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_kk_source_ready_v1.csv",
    "ru_spoof_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_ru_qwen_ready_v1.csv",
    "kk_spoof_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_kk_kazakhtts_ready_v1.csv",
    "audio_inventory": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_audio_gate_inventory_v1.csv",
    "review_packet": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_acoustic_language_packet_v1.csv",
    "reviewer_a_template": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_acoustic_language_reviewer_a_v1.csv",
    "reviewer_b_template": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_acoustic_language_reviewer_b_v1.csv",
    "materialization_receipt": "docs/artifacts/v4/xlsr_sls_model_v4_final_salvage_materialization_v1.json",
    "pair_lock_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_salvage_pairs_frozen_v1.csv",
    "pair_lock_receipt": "docs/artifacts/v4/xlsr_sls_model_v4_final_salvage_pair_lock_v1.json",
}
_REQUIRED_INPUTS = {
    "metadata_plan",
    "metadata_receipt",
    "metadata_selection",
    "recovery_selection",
    "salvage_selection",
    "permanent_rejects",
    "salvage_authorization",
    "recovery_materialization_plan",
    "recovery_failure_receipt",
    "recovery_qwen_journal",
    "recovery_kk_journal",
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
    "calibration_source_decode_journal",
    "calibration_spoof_decode_journal",
    "final_inputs_module",
    "base_materialization_module",
    "recovery_materialization_module",
    "salvage_materialization_module",
    "audio_gate_module",
    "common_voice_module",
    "fleurs_module",
    "qwen_module",
    "qwen_recovery_module",
    "kazakhtts_module",
    "kazakhtts_inference_module",
    "runner_script",
}
_PROHIBITIONS = {
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
    "reuse_permanently_rejected_kk_audio_or_text",
    "pair_lock_before_two_reviews",
}


class V4FinalSalvageError(ValueError):
    """Raised when the finite salvage boundary cannot be proven."""


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        item: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalSalvageError(f"Cannot read {label}: {path}") from error
    if not isinstance(item, dict):
        raise V4FinalSalvageError(f"{label} must be a JSON object.")
    return cast(dict[str, object], item)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4FinalSalvageError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4FinalSalvageError(f"{label} must be a non-empty project-relative path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise V4FinalSalvageError(f"{label} is not a safe project-relative path.")
    return parsed.as_posix()


def _project_path(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise V4FinalSalvageError(f"{label} escapes project root.") from error
    return path


def _verify(binding: base.Binding, root: Path, label: str) -> Path:
    try:
        return base._verify(binding, root, label)
    except base.V4FinalMaterializationError as error:
        raise V4FinalSalvageError(str(error)) from error


def _read_selection(path: Path) -> tuple[base.SelectedRow, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != V4_FINAL_SELECTION_FIELDS:
                raise V4FinalSalvageError("Salvage selection schema changed.")
            source = tuple(reader)
    except OSError as error:
        raise V4FinalSalvageError("Cannot read salvage selection.") from error
    rows: list[base.SelectedRow] = []
    for number, item in enumerate(source, start=2):
        try:
            rows.append(
                base.SelectedRow(
                    language=(item.get("language") or "").strip(),
                    selection_rank=int(item.get("selection_rank") or ""),
                    sample_id=(item.get("sample_id") or "").strip(),
                    source_member=(item.get("source_member") or "").strip(),
                    source_split=(item.get("source_split") or "").strip(),
                    parent_group_id=(item.get("parent_group_id") or "").strip(),
                    speaker_pseudo_id=(item.get("speaker_pseudo_id") or "").strip(),
                    text_id=(item.get("text_id") or "").strip(),
                    text_hash=base._sha(item.get("text_hash"), "salvage text hash"),
                    synthesis_text_sha256=base._sha(
                        item.get("synthesis_text_sha256"), "salvage synthesis hash"
                    ),
                    synthesis_seed=(item.get("synthesis_seed") or "").strip(),
                    normalization_operations=(item.get("normalization_operations") or "").strip(),
                )
            )
        except (TypeError, ValueError) as error:
            raise V4FinalSalvageError(f"Salvage selection row {number} is invalid.") from error
    return tuple(rows)


def _recovery_selection(plan: base.Plan, root: Path) -> tuple[base.SelectedRow, ...]:
    return _read_selection(_verify(plan.inputs["recovery_selection"], root, "recovery selection"))


def _load_rejects(plan: base.Plan, root: Path) -> tuple[base.SelectedRow, ...]:
    raw = _object(
        _verify(plan.inputs["permanent_rejects"], root, "permanent rejects"), "permanent rejects"
    )
    if (
        set(raw) != {"schema_version", "protocol_id", "rejections"}
        or raw["schema_version"] != 1
        or raw["protocol_id"] != PROTOCOL_ID
    ):
        raise V4FinalSalvageError("Permanent reject receipt schema is invalid.")
    values = raw["rejections"]
    if not isinstance(values, list) or len(values) != 2:
        raise V4FinalSalvageError("Salvage must record exactly two permanent rejects.")
    recovery = _recovery_selection(plan, root)
    expected: list[base.SelectedRow] = []
    for item in values:
        data = _mapping(item, "permanent rejection")
        row = next(
            (candidate for candidate in recovery if candidate.sample_id == data.get("sample_id")),
            None,
        )
        if (
            row is None
            or row.language != "kk"
            or row.selection_rank not in {272, 310}
            or data.get("selection_rank") != row.selection_rank
            or data.get("text_id") != row.text_id
            or data.get("text_hash") != row.text_hash
            or data.get("synthesis_text_sha256") != row.synthesis_text_sha256
            or data.get("resynthesis_forbidden") is not True
            or data.get("replacement_or_backfill_forbidden") is not True
            or data.get("reason") != "unsupported_character_in_locked_kazakhtts_token_list"
        ):
            raise V4FinalSalvageError("Permanent reject receipt does not bind the failed KK row.")
        expected.append(row)
    if {row.selection_rank for row in expected} != {272, 310}:
        raise V4FinalSalvageError("Permanent reject ranks changed.")
    return tuple(sorted(expected, key=lambda row: row.selection_rank))


def _load_selection(plan: base.Plan, root: Path) -> tuple[base.SelectedRow, ...]:
    recovery = _recovery_selection(plan, root)
    rejects = _load_rejects(plan, root)
    salvage = _read_selection(_verify(plan.inputs["salvage_selection"], root, "salvage selection"))
    expected = tuple(row for row in recovery if row not in rejects)
    if salvage != expected or len(salvage) != 997:
        raise V4FinalSalvageError(
            "Salvage selection must be the 999 recovery rows minus two KK rejects."
        )
    if (
        sum(row.language == "ru" for row in salvage) != 499
        or sum(row.language == "kk" for row in salvage) != 498
        or {row.selection_rank for row in salvage if row.language == "kk"}
        != set(range(1, 501)).difference({272, 310})
    ):
        raise V4FinalSalvageError("Salvage selection language/rank balance changed.")
    return salvage


@contextmanager
def _base_contract_context() -> Iterator[None]:
    replacements = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "RU_SOURCE_ID": RU_SOURCE_ID,
        "KK_SOURCE_ID": KK_SOURCE_ID,
        "RU_SPOOF_ID": RU_SPOOF_ID,
        "KK_SPOOF_ID": KK_SPOOF_ID,
    }
    prior = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(base, name, value)


def _aggregate(files: Sequence[Path], root: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    total = 0
    for path in sorted(files):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{sha256_file(path)}\0{stat.st_size}\n".encode())
        total += stat.st_size
    return len(files), total, digest.hexdigest()


def _verify_authorization(plan: base.Plan, root: Path, data_root: Path) -> None:
    authorization = _object(
        _verify(plan.inputs["salvage_authorization"], root, "salvage authorization"),
        "salvage authorization",
    )
    expected = {
        "schema_version",
        "protocol_id",
        "created_at",
        "status",
        "recovery_attempt",
        "partial_artifacts",
        "claims",
    }
    if (
        set(authorization) != expected
        or authorization["schema_version"] != 1
        or authorization["protocol_id"] != SALVAGE_AUTHORIZATION_PROTOCOL_ID
        or authorization["status"] != "partial_recovery_salvage_authorized"
    ):
        raise V4FinalSalvageError("Salvage authorization schema/status is invalid.")
    recovery_attempt = _mapping(authorization["recovery_attempt"], "recovery attempt")
    expected_bindings = {
        "plan_path": plan.inputs["recovery_materialization_plan"].path,
        "plan_sha256": plan.inputs["recovery_materialization_plan"].sha256,
        "failure_receipt_path": plan.inputs["recovery_failure_receipt"].path,
        "failure_receipt_sha256": plan.inputs["recovery_failure_receipt"].sha256,
        "qwen_journal_path": plan.inputs["recovery_qwen_journal"].path,
        "qwen_journal_sha256": plan.inputs["recovery_qwen_journal"].sha256,
        "kk_journal_path": plan.inputs["recovery_kk_journal"].path,
        "kk_journal_sha256": plan.inputs["recovery_kk_journal"].sha256,
    }
    if any(recovery_attempt.get(key) != value for key, value in expected_bindings.items()):
        raise V4FinalSalvageError("Salvage authorization does not bind the failed recovery run.")
    claims = _mapping(authorization["claims"], "salvage claims")
    if (
        claims.get("partial_outputs_reused_without_resynthesis") is not True
        or claims.get("remaining_kk_one_shot_rows") != 227
        or claims.get("permanent_kk_rejects") != 2
        or claims.get("final_inference_authorized") is not False
        or claims.get("replacement_or_backfill_authorized") is not False
    ):
        raise V4FinalSalvageError("Salvage authorization claims changed.")
    partial = _mapping(authorization["partial_artifacts"], "partial artifacts")
    if set(partial) != set(PARTIAL_RAW_ROOTS):
        raise V4FinalSalvageError("Partial artifact set changed.")
    for name, relative in PARTIAL_RAW_ROOTS.items():
        item = _mapping(partial[name], f"partial artifact {name}")
        directory = data_root / relative
        files = (
            tuple(path for path in directory.rglob("*") if path.is_file())
            if directory.is_dir()
            else ()
        )
        count, size, aggregate = _aggregate(files, directory) if directory.is_dir() else (0, 0, "")
        if item != {
            "relative_path": relative,
            "file_count": count,
            "bytes": size,
            "aggregate_sha256": aggregate,
        }:
            raise V4FinalSalvageError(f"Partial artifact evidence changed: {name}.")


def load_plan(path: Path, project_root: Path) -> base.Plan:
    root = project_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise V4FinalSalvageError("Salvage plan must be below the project root.") from error
    raw = _object(resolved, "salvage plan")
    if (
        set(raw)
        != {
            "schema_version",
            "protocol_id",
            "created_at",
            "inputs",
            "working",
            "outputs",
            "prohibitions",
        }
        or raw["schema_version"] != 1
        or raw["protocol_id"] != PROTOCOL_ID
    ):
        raise V4FinalSalvageError("Salvage plan schema/protocol is invalid.")
    raw_inputs = _mapping(raw["inputs"], "salvage inputs")
    if set(raw_inputs) != _REQUIRED_INPUTS:
        raise V4FinalSalvageError("Salvage plan inputs are incomplete.")
    inputs = {name: base._binding(value, f"inputs.{name}") for name, value in raw_inputs.items()}
    working = _mapping(raw["working"], "salvage working")
    if (
        set(working) != {"raw_roots", "processed_root", "runtime_root", "model_roots"}
        or {
            name: _safe_path(value, f"raw root {name}")
            for name, value in _mapping(working["raw_roots"], "raw roots").items()
        }
        != RAW_ROOTS
        or _safe_path(working["processed_root"], "processed root") != PROCESSED_ROOT
        or _safe_path(working["runtime_root"], "runtime root") != RUNTIME_ROOT
        or {
            name: _safe_path(value, f"model root {name}")
            for name, value in _mapping(working["model_roots"], "model roots").items()
        }
        != MODEL_ROOTS
    ):
        raise V4FinalSalvageError("Salvage namespaces or model roots changed.")
    outputs = {
        name: _safe_path(value, f"output {name}")
        for name, value in _mapping(raw["outputs"], "salvage outputs").items()
    }
    if outputs != OUTPUTS:
        raise V4FinalSalvageError("Salvage output paths changed.")
    prohibitions = _mapping(raw["prohibitions"], "salvage prohibitions")
    if set(prohibitions) != _PROHIBITIONS or any(
        value is not True for value in prohibitions.values()
    ):
        raise V4FinalSalvageError("Salvage prohibitions are not fail-closed.")
    for name, binding in inputs.items():
        _verify(binding, root, f"inputs.{name}")
    plan = base.Plan(
        relative,
        sha256_file(resolved),
        base._timestamp(raw["created_at"], "created_at"),
        inputs,
        RAW_ROOTS,
        PROCESSED_ROOT,
        RUNTIME_ROOT,
        MODEL_ROOTS,
        outputs,
    )
    _load_selection(plan, root)
    return plan


def _source_rows_from_existing(
    selected: Sequence[base.SelectedRow],
    common_voice_archive: Path,
    fleurs_root: Path,
    data_root: Path,
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
    cv_records: list[CommonVoiceRecord] = []
    cv_assets: dict[str, ExtractedCommonVoiceAsset] = {}
    texts: dict[str, str] = {}
    ru_root = data_root / PARTIAL_RAW_ROOTS["ru_source"]
    for item in ru:
        record = by_sample.get(item.sample_id)
        path = ru_root / "clips" / item.source_member
        digest = hashlib.sha256(record.sentence.encode()).hexdigest() if record else ""
        if (
            record is None
            or record.clip_name != item.source_member
            or record.sentence_id != item.text_id
            or digest != item.text_hash
            or digest != item.synthesis_text_sha256
            or f"common_voice_ru_v24:client:{record.client_id}" != item.parent_group_id
            or item.parent_group_id != item.speaker_pseudo_id
            or not path.is_file()
        ):
            raise V4FinalSalvageError(
                f"Existing RU source cannot prove selected row: {item.sample_id}."
            )
        duration, sr = inspect_extracted_common_voice_audio(path)
        cv_records.append(record)
        cv_assets[record.clip_name] = ExtractedCommonVoiceAsset(
            record.clip_name,
            path.relative_to(data_root).as_posix(),
            sha256_file(path),
            duration,
            sr,
        )
        texts[item.sample_id] = record.sentence
    ru_rows = tuple(
        replace(
            row,
            split="test",
            source_name=RU_SOURCE_ID,
            code_switch="false",
            rights_basis="Verified partial recovery Common Voice slice; personal research only; no replacement or backfill",
        )
        for row in common_voice_manifest_rows(cv_records, cv_assets, created_at=created_at)
    )
    report, by_split = inspect_fleurs_release(fleurs_root, "kk_kz")
    if report.source_splits.get("train") != 3200:
        raise V4FinalSalvageError("FLEURS kk_kz release no longer has pinned train split.")
    by_filename = {record.filename: record for record in by_split["train"]}
    kk_records: list[FleursRecord] = []
    kk_assets: dict[str, FleursExtractedAsset] = {}
    kk_root = data_root / PARTIAL_RAW_ROOTS["kk_source"]
    for item in kk:
        fleurs_record = by_filename.get(item.source_member)
        path = kk_root / item.source_member
        if (
            fleurs_record is None
            or not path.is_file()
            or f"google_fleurs_kk_v1:{fleurs_record.filename.removesuffix('.wav')}"
            != item.sample_id
            or f"google_fleurs_kk_v1:prompt:{fleurs_record.prompt_id}" != item.parent_group_id
            or item.speaker_pseudo_id != "google_fleurs_kk_v1:unknown"
            or f"google_fleurs_kk_v1:prompt:{fleurs_record.prompt_id}" != item.text_id
            or fleurs_record.text_hash != item.text_hash
            or normalize_kazakhtts_stage_c_text(fleurs_record.transcript, "kk").normalized_sha256
            != item.synthesis_text_sha256
        ):
            raise V4FinalSalvageError(
                f"Existing KK source cannot prove selected row: {item.sample_id}."
            )
        duration, sr, codec = inspect_extracted_fleurs_audio(path)
        kk_records.append(fleurs_record)
        kk_assets[fleurs_record.filename] = FleursExtractedAsset(
            fleurs_record.filename,
            path.relative_to(data_root).as_posix(),
            sha256_file(path),
            duration,
            sr,
            codec,
        )
        texts[item.sample_id] = fleurs_record.transcript
    kk_rows = tuple(
        replace(
            row,
            source_name=KK_SOURCE_ID,
            code_switch="false",
            rights_basis="Verified partial recovery FLEURS slice; CC-BY-4.0 attribution retained; no replacement or backfill",
        )
        for row in fleurs_manifest_rows(
            kk_records, kk_assets, manifest_split="test", created_at=created_at
        )
    )
    if {row.sample_id for row in ru_rows} != {row.sample_id for row in ru} or {
        row.sample_id for row in kk_rows
    } != {row.sample_id for row in kk}:
        raise V4FinalSalvageError(
            "Existing source assets do not cover exactly the salvage selection."
        )
    synthesis = {
        item.sample_id: texts[item.sample_id]
        if item.language == "ru"
        else normalize_kazakhtts_stage_c_text(texts[item.sample_id], "kk").normalized
        for item in selected
    }
    return ru_rows, kk_rows, texts, synthesis


def _read_journal(
    plan: base.Plan,
    path: Path,
    expected: Sequence[ManifestRow],
    selected: Mapping[str, base.SelectedRow],
    root: Path,
    model_name: str,
    created_at: str,
    source_id: str,
    qwen: bool,
) -> tuple[ManifestRow, ...]:
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalSalvageError(f"Cannot read partial {model_name} journal.") from error
    expected_by_id = {row.sample_id: row for row in expected}
    planned = [item for item in events if item.get("event") == "planned"]
    generated = [item for item in events if item.get("event") == "generated"]
    if (
        len(events) != 2 * len(expected)
        or len(planned) != len(expected)
        or len(generated) != len(expected)
        or {item.get("sample_id") for item in planned} != set(expected_by_id)
        or {item.get("base_sample_id") for item in generated} != set(expected_by_id)
    ):
        raise V4FinalSalvageError(
            f"Partial {model_name} journal is not an exact completed one-shot trace."
        )
    lock_name = "qwen_model_lock" if qwen else "kazakhtts_model_lock"
    lock = load_research_tts_model_lock(
        _verify(plan.inputs[lock_name], root, f"{model_name} model lock")
    )
    if len(lock.models) != 1:
        raise V4FinalSalvageError(f"Partial {model_name} lock must contain one route.")
    model = lock.models[0]
    generated_by_base = {cast(str, item["base_sample_id"]): item for item in generated}
    output: list[ManifestRow] = []
    raw_root = PARTIAL_RAW_ROOTS["ru_spoof" if qwen else "kk_spoof"]
    for item in sorted(expected, key=lambda row: selected[row.sample_id].selection_rank):
        selection = selected[item.sample_id]
        filename = (
            "ru_qwen" if qwen else "kk_kazakhtts"
        ) + f"_{selection.selection_rank:03d}_{item.text_hash[:12]}.wav"
        plan_event = next(event for event in planned if event["sample_id"] == item.sample_id)
        event = generated_by_base[item.sample_id]
        path = root / "data" / raw_root / filename
        if (
            plan_event.get("output") != filename
            or not path.is_file()
            or event.get("audio_sha256") != sha256_file(path)
        ):
            raise V4FinalSalvageError(f"Partial {model_name} WAV does not bind its journal event.")
        info = sf.info(path)
        if info.channels != 1 or info.duration <= 0:
            raise V4FinalSalvageError(f"Partial {model_name} WAV is invalid.")
        with _base_contract_context():
            row = base._spoof_row(
                item,
                model,
                source_id,
                path.relative_to(root / "data").as_posix(),
                sha256_file(path),
                float(info.duration),
                int(info.samplerate),
                created_at,
                selection.synthesis_seed,
                selection.synthesis_text_sha256,
                "cuda:0",
            )
        output.append(
            replace(row, voice_id="qwen3_tts_customvoice:aiden")
            if qwen
            else replace(
                row,
                voice_id=f"{model.model_id}:male2",
                augmentation_chain=f"text_normalizer={KAZAKHTTS_TEXT_NORMALIZER_ID};synthesis_text_sha256={selection.synthesis_text_sha256};reference_audio=forbidden;voice_cloning=false",
            )
        )
    directory = root / "data" / raw_root
    actual_files = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    expected_files = {
        (
            ("ru_qwen" if qwen else "kk_kazakhtts")
            + f"_{selected[row.sample_id].selection_rank:03d}_{row.text_hash[:12]}.wav"
        )
        for row in expected
    }
    if actual_files != expected_files:
        raise V4FinalSalvageError(
            f"Partial {model_name} raw namespace has unexpected or missing files."
        )
    validate_manifest(output)
    return tuple(output)


def _partial_rows(
    plan: base.Plan,
    root: Path,
    ru_source: Sequence[ManifestRow],
    kk_source: Sequence[ManifestRow],
    created_at: str,
) -> tuple[tuple[ManifestRow, ...], tuple[ManifestRow, ...]]:
    selected = {row.sample_id: row for row in _load_selection(plan, root)}
    ru = _read_journal(
        plan,
        _verify(plan.inputs["recovery_qwen_journal"], root, "recovery Qwen journal"),
        ru_source,
        selected,
        root,
        "Qwen",
        created_at,
        RU_SPOOF_ID,
        True,
    )
    done_kk = tuple(row for row in kk_source if selected[row.sample_id].selection_rank <= 271)
    kk = _read_journal(
        plan,
        _verify(plan.inputs["recovery_kk_journal"], root, "recovery KazakhTTS journal"),
        done_kk,
        selected,
        root,
        "KazakhTTS",
        created_at,
        KK_SPOOF_ID,
        False,
    )
    return ru, kk


def _preflight_tts(
    plan: base.Plan,
    root: Path,
    pending_texts: Mapping[str, str],
    selected: Mapping[str, base.SelectedRow],
) -> None:
    qwen = load_research_tts_model_lock(
        _verify(plan.inputs["qwen_model_lock"], root, "Qwen model lock")
    )
    if len(qwen.models) != 1:
        raise V4FinalSalvageError("Qwen lock must contain exactly one route.")
    verify_research_tts_model_lock(
        _project_path(root, plan.model_roots["qwen"], "Qwen model root"), qwen
    )
    lock = load_research_tts_model_lock(
        _verify(plan.inputs["kazakhtts_model_lock"], root, "KazakhTTS model lock")
    )
    if len(lock.models) != 1:
        raise V4FinalSalvageError("KazakhTTS lock must contain exactly one route.")
    model = lock.models[0]
    runtime = load_kazakhtts_runtime(model)
    verified = verify_research_tts_model_lock(
        _project_path(root, plan.model_roots["kazakhtts"], "KazakhTTS model root"), lock
    )
    with tempfile.TemporaryDirectory(
        prefix="kds-v4-final-salvage-kazakhtts-preflight-"
    ) as temporary:
        extracted = extract_verified_kazakhtts_runtime(
            verified_paths=verified[model.model_id],
            runtime=runtime,
            destination=Path(temporary) / "runtime",
        )
        for sample_id, text in sorted(pending_texts.items()):
            actual = validate_kazakhtts_text(text, extracted)
            if (
                actual != text
                or hashlib.sha256(text.encode()).hexdigest()
                != selected[sample_id].synthesis_text_sha256
            ):
                raise V4FinalSalvageError("Pending KazakhTTS text diverges from frozen selection.")
        load_kazakhtts_models(runtime, extracted, resolve_kazakhtts_device("cuda"))


def _require_ledger(plan: base.Plan, root: Path) -> None:
    ledger = load_license_ledger(
        _verify(plan.inputs["materialization_ledger"], root, "salvage ledger")
    )
    if set(ledger) != {RU_SOURCE_ID, KK_SOURCE_ID, RU_SPOOF_ID, KK_SPOOF_ID}:
        raise V4FinalSalvageError("Salvage ledger has an unexpected source set.")
    with _base_contract_context():
        try:
            base._require_ledger(plan, root)
        except base.V4FinalMaterializationError as error:
            raise V4FinalSalvageError(str(error)) from error


def preflight_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
    created_at: str,
) -> base.Plan:
    base._timestamp(created_at, "created_at")
    root = project_root.resolve(strict=True)
    plan = load_plan(plan_path, root)
    if data_root.resolve(strict=True) != _project_path(root, "data", "data root"):
        raise V4FinalSalvageError("data_root must be the project data directory.")
    _require_ledger(plan, root)
    try:
        base._verify_common_voice_archive(common_voice_archive)
    except base.V4FinalMaterializationError as error:
        raise V4FinalSalvageError(str(error)) from error
    selected = _load_selection(plan, root)
    _verify_authorization(plan, root, data_root)
    ru, kk, _texts, synthesis = _source_rows_from_existing(
        selected, common_voice_archive, fleurs_release_root, data_root, created_at
    )
    _partial_rows(plan, root, ru, kk, created_at)
    selected_by_id = {row.sample_id: row for row in selected}
    pending = {
        row.sample_id: synthesis[row.sample_id]
        for row in kk
        if selected_by_id[row.sample_id].selection_rank > 271
    }
    if len(pending) != 227:
        raise V4FinalSalvageError("Salvage pending KazakhTTS scope must contain exactly 227 rows.")
    with _base_contract_context():
        try:
            base._history(plan, root)
        except base.V4FinalMaterializationError as error:
            raise V4FinalSalvageError(str(error)) from error
    runtime = _project_path(root, plan.runtime_root, "salvage runtime")
    if runtime.exists() or (data_root / plan.raw_roots["kk_spoof"]).exists():
        raise V4FinalSalvageError("Salvage one-shot namespace already exists.")
    for relative in plan.outputs.values():
        target = _project_path(root, relative, "salvage output")
        if target.exists() or not target.parent.is_dir():
            raise V4FinalSalvageError("Every salvage output must be new with an existing parent.")
    _preflight_tts(plan, root, pending, {row.sample_id: row for row in selected})
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
) -> base.Plan:
    if workers <= 0:
        raise V4FinalSalvageError("workers must be positive.")
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
    selected_by_id = {row.sample_id: row for row in selected}
    ru_source, kk_source, source_text, synthesis_text = _source_rows_from_existing(
        selected, common_voice_archive, fleurs_release_root, data_root, created_at
    )
    ru_spoof, partial_kk = _partial_rows(plan, root, ru_source, kk_source, created_at)
    pending_source = tuple(
        row for row in kk_source if selected_by_id[row.sample_id].selection_rank > 271
    )
    runtime = _project_path(root, plan.runtime_root, "salvage runtime")
    runtime.mkdir(parents=True)
    with _base_contract_context():
        try:
            generated_kk = base._synthesize_kk(
                pending_source, source_text, selected_by_id, plan, root, data_root, created_at
            )
        except base.V4FinalMaterializationError as error:
            raise V4FinalSalvageError(str(error)) from error
    kk_spoof = tuple(
        sorted(
            (*partial_kk, *generated_kk),
            key=lambda row: (
                selected_by_id[
                    next(source.sample_id for source in kk_source if source.text_id == row.text_id)
                ].selection_rank
            ),
        )
    )
    groups = {
        "ru_source": ru_source,
        "kk_source": kk_source,
        "ru_spoof": ru_spoof,
        "kk_spoof": kk_spoof,
    }
    ledger = load_license_ledger(
        _verify(plan.inputs["materialization_ledger"], root, "salvage ledger")
    )
    for rows in groups.values():
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
        require_valid_assets(rows, data_root)
    decoded = {
        name: _decode(rows, name, plan, data_root, runtime, workers)
        for name, rows in groups.items()
    }
    with _base_contract_context():
        historical_exact, historical_signatures, history = base._history(plan, root)
        ranks = {row.sample_id: row.selection_rank for row in selected}
        pairs = {row.sample_id: row.pair_key for row in selected}
        for source_rows, spoof_rows in ((ru_source, ru_spoof), (kk_source, kk_spoof)):
            by_text = {row.text_id: row for row in source_rows}
            for spoof in spoof_rows:
                source = by_text.get(spoof.text_id)
                if source is None or source.text_hash != spoof.text_hash:
                    raise V4FinalSalvageError("Salvage spoof route broke source text pairing.")
                ranks[spoof.sample_id] = ranks[source.sample_id]
                pairs[spoof.sample_id] = pairs[source.sample_id]
        decisions = {
            name: base._decisions(
                rows, decoded[name], ranks, historical_exact, historical_signatures
            )
            for name, rows in groups.items()
        }
        ready = {
            name: base._ready({row.sample_id: row for row in rows}, decisions[name], created_at)
            for name, rows in groups.items()
        }
    all_ready = tuple(
        row for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof") for row in ready[name]
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
        decisions_all = tuple(
            item
            for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof")
            for item in decisions[name]
        )
        base._write_csv(
            staged["audio_inventory"],
            base.INVENTORY_FIELDS,
            base._inventory(decisions_all, pairs, historical_exact),
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
                "selected_source_rows": {"ru": len(ru_source), "kk": len(kk_source)},
                "partial_reused_synthetic_rows": {"ru": len(ru_spoof), "kk": len(partial_kk)},
                "new_one_shot_synthetic_rows": {"ru": 0, "kk": len(generated_kk)},
                "permanent_rejected_original_rows": {"ru": 1, "kk": 2},
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
                    "rows": base._rows(staged[name]),
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
                "raw_audio_extraction_performed": False,
                "partial_recovery_assets_reused_without_resynthesis": True,
                "new_remaining_kk_synthesis_performed": True,
                "technical_decode_qa_vad_performed": True,
                "full_history_audio_isolation_performed": True,
                "acoustic_review_performed": False,
                "pair_lock_performed": False,
                "detector_checkpoint_loaded": False,
                "calibration_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "final_inference_performed": False,
                "final_inference_authorized": False,
                "replacement_or_backfill": False,
                "resynthesis": False,
            },
        }
        staged["materialization_receipt"].write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with _base_contract_context():
            packet = base._review_packet(
                all_ready,
                sha256_file(staged["materialization_receipt"]),
                source_text,
                synthesis_text,
                pairs,
                data_root,
            )
        base._write_csv(
            staged["review_packet"], base.PACKET_FIELDS, [asdict(row) for row in packet]
        )
        packet_sha = sha256_file(staged["review_packet"])
        for name, reviewer in (
            ("reviewer_a_template", "reviewer_A_REPLACE_ME"),
            ("reviewer_b_template", "reviewer_B_REPLACE_ME"),
        ):
            base._write_csv(
                staged[name],
                base.REVIEW_FIELDS,
                [
                    {
                        "protocol_id": PROTOCOL_ID,
                        "packet_sha256": packet_sha,
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
        publish = tuple(
            name for name in plan.outputs if name not in {"pair_lock_manifest", "pair_lock_receipt"}
        )
        if any(
            _project_path(root, plan.outputs[name], "salvage output").exists() for name in publish
        ):
            raise V4FinalSalvageError("A salvage output appeared during staging.")
        for name in publish:
            staged[name].replace(_project_path(root, plan.outputs[name], "salvage output"))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return plan


def finalize_pair_lock(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    reviewer_a: Path,
    reviewer_b: Path,
    created_at: str,
) -> base.Plan:
    """Lock only genuine two-review passes; final inference remains unavailable."""
    with _base_contract_context():
        old_plan, old_selection = base.load_plan, base._load_selection
        try:
            base.load_plan, base._load_selection = load_plan, _load_selection
            return base.finalize_pair_lock(
                plan_path=plan_path,
                project_root=project_root,
                data_root=data_root,
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                created_at=created_at,
            )
        finally:
            base.load_plan, base._load_selection = old_plan, old_selection
