"""Strict one-time XLS-R evaluation contract for the locked ToneSpeak spoof-only OOD set.

This contract is intentionally separate from the mixed stress-test runner.  ToneSpeak has no
bona-fide class, so its single permitted run can report only spoof recall at the frozen model's
raw zero-logit boundary.  It cannot produce accuracy, balanced accuracy, calibration, a selected
threshold, or a final/product claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.licenses import LicenseLedgerEntry, LicenseLedgerError, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.eval.metrics import WilsonInterval, wilson_interval
from kds.eval.tone_speak_acoustic_gate import (
    TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID,
    read_tone_speak_acoustic_packet,
)
from kds.training.xlsr_stage_a_plan import PinnedFile

TONE_SPEAK_OOD_RUN_SCHEMA_VERSION = 1
TONE_SPEAK_OOD_SOURCE_ID = "tone_speak_ru_v1"
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_PROTOCOL = {
    "kind": "exploratory_russian_spoof_only_ood_evaluation",
    "quality_claim": "not_final_quality",
    "training": "prohibited",
    "calibration": "prohibited",
    "threshold_selection": "prohibited",
    "binary_metrics": "unavailable_spoof_only",
    "acoustic_language_preservation": "verified_for_pinned_assets_only",
}


class ToneSpeakOodPlanError(ValueError):
    """Raised when the research-only ToneSpeak OOD contract cannot be trusted."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class ToneSpeakFrozenCheckpoint:
    checkpoint: PinnedFile
    report: PinnedFile
    selected_trainable_state_sha256: str


@dataclass(frozen=True, slots=True)
class ToneSpeakPinnedXlsrEncoder:
    checkpoint_dir: Path
    revision: str
    config: PinnedFile
    weights: PinnedFile


@dataclass(frozen=True, slots=True)
class ToneSpeakXlsrSlsHead:
    attention_size: int
    classifier_size: int
    dropout: float


@dataclass(frozen=True, slots=True)
class ToneSpeakOodCandidate:
    manifest: PinnedFile
    ready_receipt: PinnedFile
    acoustic_gate_packet: PinnedFile
    acoustic_gate_report: PinnedFile
    source_audit_receipt: PinnedFile
    source_artifact_lock: PinnedFile
    expected_rows: int
    expected_source_id: str
    expected_voice_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToneSpeakOodInferenceConfig:
    sample_rate: int
    window_samples: int
    batch_size: int
    num_workers: int
    device: str
    precision: str
    raw_logit_decision_boundary: float


@dataclass(frozen=True, slots=True)
class ToneSpeakOodOutputs:
    execution_lock: Path
    report: Path


@dataclass(frozen=True, slots=True)
class ToneSpeakOodPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    protocol: dict[str, str]
    license_ledger: PinnedFile
    checkpoint: ToneSpeakFrozenCheckpoint
    encoder: ToneSpeakPinnedXlsrEncoder
    head: ToneSpeakXlsrSlsHead
    candidate: ToneSpeakOodCandidate
    implementation: tuple[PinnedFile, ...]
    inference: ToneSpeakOodInferenceConfig
    outputs: ToneSpeakOodOutputs


def load_tone_speak_ood_plan(path: Path) -> ToneSpeakOodPlan:
    """Load a plan and verify every static byte before a CUDA model can be constructed."""

    if not path.is_file():
        raise ToneSpeakOodPlanError(f"ToneSpeak OOD run plan does not exist: {path}")
    try:
        plan_bytes = path.read_bytes()
        raw_value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ToneSpeakOodPlanError(
            f"Cannot read ToneSpeak OOD run plan {path}: {error}"
        ) from error
    raw = _object(raw_value, "ToneSpeak OOD run plan")
    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "purpose",
            "protocol",
            "license_ledger",
            "checkpoint",
            "encoder",
            "head",
            "candidate",
            "implementation",
            "inference",
            "outputs",
        },
        "ToneSpeak OOD run plan",
    )
    if raw["schema_version"] != TONE_SPEAK_OOD_RUN_SCHEMA_VERSION:
        raise ToneSpeakOodPlanError("ToneSpeak OOD run plan schema_version must be 1.")
    if _required_string(raw, "purpose", "ToneSpeak OOD run plan") != "research":
        raise ToneSpeakOodPlanError("ToneSpeak OOD plans support purpose='research' only.")
    run_id = _required_string(raw, "run_id", "ToneSpeak OOD run plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise ToneSpeakOodPlanError("run_id contains unsupported characters.")
    base = path.resolve().parent
    protocol = _parse_protocol(raw["protocol"])
    plan = ToneSpeakOodPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        protocol=protocol,
        license_ledger=_parse_pinned_file(raw["license_ledger"], "license_ledger", base),
        checkpoint=_parse_checkpoint(raw["checkpoint"], base),
        encoder=_parse_encoder(raw["encoder"], base),
        head=_parse_head(raw["head"]),
        candidate=_parse_candidate(raw["candidate"], base),
        implementation=_parse_implementation(raw["implementation"], base),
        inference=_parse_inference(raw["inference"]),
        outputs=_parse_outputs(raw["outputs"], base),
    )
    if plan.outputs.execution_lock == plan.outputs.report:
        raise ToneSpeakOodPlanError("execution_lock and report paths must differ.")
    for pinned, label in (
        (plan.license_ledger, "License ledger"),
        (plan.checkpoint.checkpoint, "Frozen Stage-B checkpoint"),
        (plan.checkpoint.report, "Frozen Stage-B report"),
        (plan.encoder.config, "XLS-R config"),
        (plan.encoder.weights, "XLS-R weights"),
        (plan.candidate.manifest, "ToneSpeak OOD manifest"),
        (plan.candidate.ready_receipt, "ToneSpeak ready receipt"),
        (plan.candidate.acoustic_gate_packet, "ToneSpeak acoustic gate packet"),
        (plan.candidate.acoustic_gate_report, "ToneSpeak acoustic gate report"),
        (plan.candidate.source_audit_receipt, "ToneSpeak source audit receipt"),
        (plan.candidate.source_artifact_lock, "ToneSpeak source artifact lock"),
        *((item, "Pinned implementation") for item in plan.implementation),
    ):
        _verify_pinned_file(pinned, label)
    _validate_stage_b_report(
        plan.checkpoint.report.path, plan.checkpoint.selected_trainable_state_sha256
    )
    if not plan.outputs.execution_lock.parent.is_dir() or not plan.outputs.report.parent.is_dir():
        raise ToneSpeakOodPlanError("ToneSpeak OOD output parent directories must already exist.")
    return plan


def validate_tone_speak_ood_inputs(
    plan: ToneSpeakOodPlan, ledger: Mapping[str, LicenseLedgerEntry]
) -> tuple[ManifestRow, ...]:
    """Verify that the planned data are exactly the reviewed 100-row spoof-only OOD candidate."""

    try:
        rows = load_manifest(plan.candidate.manifest.path)
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
    except (ManifestError, LicenseLedgerError) as error:
        raise ToneSpeakOodPlanError(error.issues) from error
    voice_counts = Counter(row.voice_id for row in rows)
    issues: list[str] = []
    if len(rows) != plan.candidate.expected_rows:
        issues.append(f"ToneSpeak OOD candidate must contain {plan.candidate.expected_rows} rows.")
    if len({row.sample_id for row in rows}) != len(rows):
        issues.append("ToneSpeak OOD candidate has duplicate sample IDs.")
    if len({row.text_hash for row in rows}) != len(rows):
        issues.append("ToneSpeak OOD candidate has duplicate text groups.")
    if Counter(row.label for row in rows) != Counter({"spoof": plan.candidate.expected_rows}):
        issues.append("ToneSpeak OOD candidate must be spoof-only; binary metrics are unavailable.")
    if {row.source_name for row in rows} != {plan.candidate.expected_source_id}:
        issues.append("ToneSpeak OOD candidate source differs from the pinned plan.")
    if any(
        row.split != "ood"
        or row.language != "ru"
        or row.code_switch != "false"
        or row.codec != "wav"
        or row.generator_name != "openai_gpt_4o_mini_tts"
        or row.generator_version != "source_card_unpinned"
        for row in rows
    ):
        issues.append("ToneSpeak OOD rows must be ready RU spoof-only source-card observations.")
    expected_voice_counts = Counter({voice: 10 for voice in plan.candidate.expected_voice_ids})
    if voice_counts != expected_voice_counts:
        issues.append("ToneSpeak OOD voice counts differ from the locked ten-voice selection.")
    _validate_ready_receipt(plan.candidate.ready_receipt.path, plan)
    _validate_source_receipts(
        plan.candidate.source_audit_receipt.path, plan.candidate.source_artifact_lock.path
    )
    _validate_acoustic_gate(plan, rows, issues)
    if issues:
        raise ToneSpeakOodPlanError(issues)
    return tuple(rows)


def tone_speak_ood_plan_record(plan: ToneSpeakOodPlan) -> dict[str, object]:
    """Return the complete plan record for a write-once execution receipt."""

    return {
        "schema_version": TONE_SPEAK_OOD_RUN_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.plan_path),
        "plan_sha256": plan.plan_sha256,
        "protocol": dict(plan.protocol),
        "license_ledger": _pinned_record(plan.license_ledger),
        "checkpoint": {
            "path": str(plan.checkpoint.checkpoint.path),
            "sha256": plan.checkpoint.checkpoint.sha256,
            "stage_b_report": _pinned_record(plan.checkpoint.report),
            "selected_trainable_state_sha256": plan.checkpoint.selected_trainable_state_sha256,
        },
        "encoder": {
            "checkpoint_dir": str(plan.encoder.checkpoint_dir),
            "revision": plan.encoder.revision,
            "config": _pinned_record(plan.encoder.config),
            "weights": _pinned_record(plan.encoder.weights),
        },
        "head": asdict(plan.head),
        "candidate": {
            "manifest": _pinned_record(plan.candidate.manifest),
            "ready_receipt": _pinned_record(plan.candidate.ready_receipt),
            "acoustic_gate_packet": _pinned_record(plan.candidate.acoustic_gate_packet),
            "acoustic_gate_report": _pinned_record(plan.candidate.acoustic_gate_report),
            "source_audit_receipt": _pinned_record(plan.candidate.source_audit_receipt),
            "source_artifact_lock": _pinned_record(plan.candidate.source_artifact_lock),
            "expected_rows": plan.candidate.expected_rows,
            "expected_source_id": plan.candidate.expected_source_id,
            "expected_voice_ids": list(plan.candidate.expected_voice_ids),
        },
        "implementation": [_pinned_record(item) for item in plan.implementation],
        "inference": asdict(plan.inference),
        "outputs": {
            "execution_lock": str(plan.outputs.execution_lock),
            "report": str(plan.outputs.report),
        },
    }


def metric_record(correct: int, examples: int) -> dict[str, object]:
    """Serialize a fixed-boundary count with a Wilson interval; this is not calibration."""

    if examples <= 0 or correct < 0 or correct > examples:
        raise ValueError("Metric counts are invalid.")
    interval: WilsonInterval = wilson_interval(correct, examples)
    return {
        "correct": correct,
        "examples": examples,
        "value": correct / examples,
        "confidence_interval": asdict(interval),
    }


def _parse_protocol(value: object) -> dict[str, str]:
    raw = _object(value, "protocol")
    _expect_exact_keys(raw, set(_PROTOCOL), "protocol")
    parsed = {key: _required_string(raw, key, "protocol") for key in _PROTOCOL}
    if parsed != _PROTOCOL:
        raise ToneSpeakOodPlanError("ToneSpeak OOD protocol must retain all research-only limits.")
    return parsed


def _parse_checkpoint(value: object, base: Path) -> ToneSpeakFrozenCheckpoint:
    raw = _object(value, "checkpoint")
    _expect_exact_keys(
        raw,
        {"path", "sha256", "stage_b_report", "selected_trainable_state_sha256"},
        "checkpoint",
    )
    return ToneSpeakFrozenCheckpoint(
        checkpoint=PinnedFile(
            path=_relative_path(raw, "path", "checkpoint", base),
            sha256=_required_sha256(raw, "sha256", "checkpoint"),
        ),
        report=_parse_pinned_file(raw["stage_b_report"], "stage_b_report", base),
        selected_trainable_state_sha256=_required_sha256(
            raw, "selected_trainable_state_sha256", "checkpoint"
        ),
    )


def _parse_encoder(value: object, base: Path) -> ToneSpeakPinnedXlsrEncoder:
    raw = _object(value, "encoder")
    _expect_exact_keys(raw, {"checkpoint_dir", "revision", "config", "weights"}, "encoder")
    checkpoint_dir = _relative_path(raw, "checkpoint_dir", "encoder", base)
    if not checkpoint_dir.is_dir():
        raise ToneSpeakOodPlanError(
            f"Pinned XLS-R checkpoint directory does not exist: {checkpoint_dir}"
        )
    revision = _required_string(raw, "revision", "encoder")
    if _GIT_REVISION.fullmatch(revision) is None:
        raise ToneSpeakOodPlanError("encoder revision must be a 40- or 64-character hash.")
    return ToneSpeakPinnedXlsrEncoder(
        checkpoint_dir=checkpoint_dir,
        revision=revision,
        config=_parse_pinned_file(raw["config"], "encoder config", base),
        weights=_parse_pinned_file(raw["weights"], "encoder weights", base),
    )


def _parse_head(value: object) -> ToneSpeakXlsrSlsHead:
    raw = _object(value, "head")
    _expect_exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    return ToneSpeakXlsrSlsHead(
        attention_size=_required_int(raw, "attention_size", "head", minimum=1),
        classifier_size=_required_int(raw, "classifier_size", "head", minimum=1),
        dropout=_required_float(raw, "dropout", "head", minimum=0.0),
    )


def _parse_candidate(value: object, base: Path) -> ToneSpeakOodCandidate:
    raw = _object(value, "candidate")
    _expect_exact_keys(
        raw,
        {
            "manifest",
            "ready_receipt",
            "acoustic_gate_packet",
            "acoustic_gate_report",
            "source_audit_receipt",
            "source_artifact_lock",
            "expected_rows",
            "expected_source_id",
            "expected_voice_ids",
        },
        "candidate",
    )
    voices = _string_tuple(raw["expected_voice_ids"], "candidate expected_voice_ids")
    if len(voices) != 10 or voices != tuple(sorted(voices)) or len(set(voices)) != len(voices):
        raise ToneSpeakOodPlanError("candidate expected_voice_ids must be ten sorted unique IDs.")
    rows = _required_int(raw, "expected_rows", "candidate", minimum=1)
    if rows != 100:
        raise ToneSpeakOodPlanError("ToneSpeak OOD candidate expected_rows must be exactly 100.")
    source = _required_string(raw, "expected_source_id", "candidate")
    if source != TONE_SPEAK_OOD_SOURCE_ID:
        raise ToneSpeakOodPlanError("ToneSpeak OOD candidate source must be tone_speak_ru_v1.")
    return ToneSpeakOodCandidate(
        manifest=_parse_pinned_file(raw["manifest"], "candidate manifest", base),
        ready_receipt=_parse_pinned_file(raw["ready_receipt"], "candidate ready_receipt", base),
        acoustic_gate_packet=_parse_pinned_file(
            raw["acoustic_gate_packet"], "candidate acoustic_gate_packet", base
        ),
        acoustic_gate_report=_parse_pinned_file(
            raw["acoustic_gate_report"], "candidate acoustic_gate_report", base
        ),
        source_audit_receipt=_parse_pinned_file(
            raw["source_audit_receipt"], "candidate source_audit_receipt", base
        ),
        source_artifact_lock=_parse_pinned_file(
            raw["source_artifact_lock"], "candidate source_artifact_lock", base
        ),
        expected_rows=rows,
        expected_source_id=source,
        expected_voice_ids=voices,
    )


def _parse_implementation(value: object, base: Path) -> tuple[PinnedFile, ...]:
    if not isinstance(value, list) or not value:
        raise ToneSpeakOodPlanError("implementation must be a non-empty array.")
    pinned = tuple(
        _parse_pinned_file(item, f"implementation {index}", base)
        for index, item in enumerate(value, start=1)
    )
    if len({item.path for item in pinned}) != len(pinned):
        raise ToneSpeakOodPlanError("implementation must not repeat a file path.")
    return pinned


def _parse_inference(value: object) -> ToneSpeakOodInferenceConfig:
    raw = _object(value, "inference")
    _expect_exact_keys(
        raw,
        {
            "sample_rate",
            "window_samples",
            "batch_size",
            "num_workers",
            "device",
            "precision",
            "raw_logit_decision_boundary",
        },
        "inference",
    )
    device = _required_string(raw, "device", "inference")
    precision = _required_string(raw, "precision", "inference")
    boundary = _required_float(raw, "raw_logit_decision_boundary", "inference")
    if device != "cuda" or precision != "bf16" or boundary != 0.0:
        raise ToneSpeakOodPlanError(
            "ToneSpeak OOD inference requires CUDA BF16 and the fixed zero raw-logit boundary."
        )
    return ToneSpeakOodInferenceConfig(
        sample_rate=_required_int(raw, "sample_rate", "inference", minimum=1),
        window_samples=_required_int(raw, "window_samples", "inference", minimum=1),
        batch_size=_required_int(raw, "batch_size", "inference", minimum=1),
        num_workers=_required_int(raw, "num_workers", "inference", minimum=0),
        device=device,
        precision=precision,
        raw_logit_decision_boundary=boundary,
    )


def _parse_outputs(value: object, base: Path) -> ToneSpeakOodOutputs:
    raw = _object(value, "outputs")
    _expect_exact_keys(raw, {"execution_lock", "report"}, "outputs")
    return ToneSpeakOodOutputs(
        execution_lock=_relative_path(raw, "execution_lock", "outputs", base),
        report=_relative_path(raw, "report", "outputs", base),
    )


def _validate_ready_receipt(path: Path, plan: ToneSpeakOodPlan) -> None:
    receipt = _load_json_object(path, "ToneSpeak ready receipt")
    if (
        receipt.get("source_id") != plan.candidate.expected_source_id
        or receipt.get("ready_manifest_sha256") != plan.candidate.manifest.sha256
        or receipt.get("ready_rows") != plan.candidate.expected_rows
        or receipt.get("raw_rows") != plan.candidate.expected_rows
        or receipt.get("rejected_rows") != 0
        or receipt.get("final_or_product_eligible") is not False
    ):
        raise ToneSpeakOodPlanError(
            "ToneSpeak ready receipt does not bind the locked OOD manifest."
        )


def _validate_source_receipts(audit_path: Path, lock_path: Path) -> None:
    audit = _load_json_object(audit_path, "ToneSpeak source audit receipt")
    lock = _load_json_object(lock_path, "ToneSpeak source artifact lock")
    if (
        audit.get("source_id") != TONE_SPEAK_OOD_SOURCE_ID
        or audit.get("revision") != lock.get("revision")
        or lock.get("source_id") != TONE_SPEAK_OOD_SOURCE_ID
        or audit.get("rows_by_split") != {"train": 6298, "validation": 700}
        or audit.get("audio_records") != 6998
    ):
        raise ToneSpeakOodPlanError(
            "ToneSpeak source audit/lock does not match the accepted release."
        )


def _validate_acoustic_gate(
    plan: ToneSpeakOodPlan, rows: list[ManifestRow], issues: list[str]
) -> None:
    try:
        packet = read_tone_speak_acoustic_packet(plan.candidate.acoustic_gate_packet.path)
    except (OSError, ToneSpeakOodPlanError, ValueError) as error:
        issues.append(f"Cannot validate ToneSpeak acoustic gate packet: {error}")
        return
    packet_by_id = {item.sample_id: item for item in packet}
    if len(packet_by_id) != plan.candidate.expected_rows:
        issues.append("ToneSpeak acoustic packet does not contain 100 unique records.")
    for row in rows:
        packet_row = packet_by_id.get(row.sample_id)
        if (
            packet_row is None
            or packet_row.audio_sha256 != row.sha256
            or packet_row.text_hash != row.text_hash
            or packet_row.relative_path != row.relative_path
        ):
            issues.append("ToneSpeak acoustic packet does not bind the exact ready OOD manifest.")
            break
    try:
        report = _load_json_object(
            plan.candidate.acoustic_gate_report.path, "ToneSpeak acoustic gate report"
        )
    except ToneSpeakOodPlanError as error:
        issues.extend(error.issues)
        return
    report_packet = report.get("packet")
    review_files = report.get("review_files")
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id") != TONE_SPEAK_ACOUSTIC_GATE_PROTOCOL_ID
        or report.get("all_assets_acoustically_verified") is not True
        or report.get("final_or_product_eligible") is not False
        or report.get("review_rows") != 200
        or report.get("decision_counts") != {"pass": plan.candidate.expected_rows}
        or not isinstance(report_packet, dict)
        or report_packet.get("sha256") != plan.candidate.acoustic_gate_packet.sha256
        or report_packet.get("assets") != plan.candidate.expected_rows
        or not isinstance(review_files, list)
        or len(review_files) != 2
    ):
        issues.append("ToneSpeak acoustic gate report is incomplete or not eligible for this run.")
        return
    reviewer_ids = {
        item.get("reviewer_pseudo_id")
        for item in review_files
        if isinstance(item, dict)
        and item.get("rows") == plan.candidate.expected_rows
        and isinstance(item.get("sha256"), str)
        and _SHA256.fullmatch(item["sha256"]) is not None
    }
    if len(reviewer_ids) != 2 or not all(isinstance(item, str) and item for item in reviewer_ids):
        issues.append(
            "ToneSpeak acoustic gate report does not pin two distinct complete review files."
        )


def _validate_stage_b_report(path: Path, state_sha256: str) -> None:
    report = _load_json_object(path, "Pinned Stage-B report")
    if (
        report.get("status") != "ok"
        or report.get("checkpoint_scope") != "sls_head_and_final_xlsr_blocks"
        or report.get("frozen_final_evaluation_performed") is not False
        or report.get("calibrated") is not False
        or report.get("selected_trainable_state_sha256") != state_sha256
    ):
        raise ToneSpeakOodPlanError(
            "Pinned Stage-B report is not the expected uncalibrated frozen Stage-B receipt."
        )


def _parse_pinned_file(value: object, label: str, base: Path) -> PinnedFile:
    raw = _object(value, label)
    return _pinned_from_path_fields(raw, label, base)


def _pinned_from_path_fields(raw: Mapping[str, object], label: str, base: Path) -> PinnedFile:
    _expect_exact_keys(raw, {"path", "sha256"}, label)
    return PinnedFile(
        path=_relative_path(raw, "path", label, base),
        sha256=_required_sha256(raw, "sha256", label),
    )


def _verify_pinned_file(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file():
        raise ToneSpeakOodPlanError(f"{label} does not exist: {pinned.path}")
    try:
        actual = sha256_file(pinned.path)
    except OSError as error:
        raise ToneSpeakOodPlanError(f"Cannot hash {label}: {error}") from error
    if actual != pinned.sha256:
        raise ToneSpeakOodPlanError(
            f"{label} SHA-256 mismatch: expected {pinned.sha256}, got {actual}."
        )


def _pinned_record(pinned: PinnedFile) -> dict[str, str]:
    return {"path": str(pinned.path), "sha256": pinned.sha256}


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ToneSpeakOodPlanError(f"Cannot read {label} {path}: {error}") from error
    return _object(value, label)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ToneSpeakOodPlanError(f"{label} must be a JSON object.")
    return value


def _expect_exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    found = set(raw)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ToneSpeakOodPlanError(
            f"{label} fields mismatch; missing={missing!r}, extra={extra!r}."
        )


def _required_string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise ToneSpeakOodPlanError(f"{label} field {name!r} must be a non-empty string.")
    return value


def _required_sha256(raw: Mapping[str, object], name: str, label: str) -> str:
    value = _required_string(raw, name, label)
    if _SHA256.fullmatch(value) is None:
        raise ToneSpeakOodPlanError(f"{label} field {name!r} must be a lowercase SHA-256.")
    return value


def _required_int(raw: Mapping[str, object], name: str, label: str, minimum: int) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ToneSpeakOodPlanError(f"{label} field {name!r} must be an integer >= {minimum}.")
    return value


def _required_float(
    raw: Mapping[str, object], name: str, label: str, minimum: float | None = None
) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToneSpeakOodPlanError(f"{label} field {name!r} must be a finite number.")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        minimum_text = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ToneSpeakOodPlanError(f"{label} field {name!r} must be {minimum_text}.")
    return parsed


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ToneSpeakOodPlanError(f"{label} must be a non-empty array of strings.")
    return tuple(value)


def _relative_path(raw: Mapping[str, object], name: str, label: str, base: Path) -> Path:
    value = _required_string(raw, name, label)
    candidate = Path(value)
    if candidate.is_absolute():
        raise ToneSpeakOodPlanError(f"{label} field {name!r} must be a relative path.")
    return (base / candidate).resolve()
