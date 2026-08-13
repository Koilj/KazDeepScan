#!/usr/bin/env python3
"""Execute the one immutable v3 XLS-R evaluation on the frozen Stage-D RU pairs.

This command intentionally reuses only computation helpers from the archived v2 evaluator.
It never opens the v2 inference receipt, logits, errors, or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from evaluate_xlsr_stage_d_dialogs_ru import (
    Checkpoint,
    Encoder,
    Head,
    Inference,
    PinnedFile,
    _build_model,
    _final_report,
    _infer_logits,
    _load_stage_b_state,
)

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerEntry, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.stage_b_dev import STAGE_B_LEAKAGE_FIELDS
from kds.eval.calibration import TemperatureScaler

SCHEMA_VERSION = 1
PROTOCOL_ID = "xlsr-sls-v3-stage-d-dialogs-ru-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_PROTOCOL = {
    "kind": "governed_one_time_v3_stage_d_ru_research_evaluation",
    "quality_claim": "research_only_not_product_quality",
    "stage_d_history": (
        "exact_assets_were_evaluated_by_v2_but_v2_logits_errors_unloaded_and_"
        "prohibited_for_v3_selection"
    ),
    "checkpoint_selection": "stage_a_and_stage_b_dev_loss_only",
    "calibration": "temperature_only_on_pinned_pyara_role",
    "decision_boundary": "fixed_calibrated_probability_0.5",
    "pooled_language_metric": "prohibited",
    "final_set_model_driven_selection": "prohibited_no_reselection_no_backfill",
    "one_time_final_inference": "required_after_write_once_preflight",
}


class V3StageDEvaluationError(ValueError):
    """Raised when an immutable v3 evaluation input is not trustworthy."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class LabelCounts:
    bonafide: int
    spoof: int

    def as_dict(self) -> dict[str, int]:
        return {"bonafide": self.bonafide, "spoof": self.spoof}


@dataclass(frozen=True, slots=True)
class CalibrationRole:
    manifest: PinnedFile
    expected_rows: int
    expected_labels: LabelCounts


@dataclass(frozen=True, slots=True)
class FinalRuRole:
    manifest: PinnedFile
    expected_rows: int
    expected_pairs: int
    expected_labels: LabelCounts
    acoustic_gate: PinnedFile
    pairing_receipt: PinnedFile
    project_exposure_audit: PinnedFile
    route_audit: PinnedFile


@dataclass(frozen=True, slots=True)
class Governance:
    contract: PinnedFile
    receipt: PinnedFile


@dataclass(frozen=True, slots=True)
class V3Checkpoint:
    checkpoint: Checkpoint
    stage_b_plan: PinnedFile


@dataclass(frozen=True, slots=True)
class Outputs:
    preflight: Path
    execution_lock: Path
    report: Path


@dataclass(frozen=True, slots=True)
class Plan:
    run_id: str
    path: Path
    sha256: str
    protocol: dict[str, str]
    license_ledger: PinnedFile
    governance: Governance
    checkpoint: V3Checkpoint
    encoder: Encoder
    head: Head
    calibration: CalibrationRole
    final_ru: FinalRuRole
    implementation: tuple[PinnedFile, ...]
    inference: Inference
    outputs: Outputs


@dataclass(frozen=True, slots=True)
class Inputs:
    calibration: tuple[ManifestRow, ...]
    final_ru: tuple[ManifestRow, ...]


@dataclass(frozen=True, slots=True)
class ComputePlan:
    """The minimal frozen interface consumed by audited v2 computation helpers."""

    run_id: str
    checkpoint: Checkpoint
    encoder: Encoder
    head: Head
    inference: Inference


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V3StageDEvaluationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        raise V3StageDEvaluationError(
            f"{label} fields differ; missing={missing!r}, unknown={unknown!r}."
        )


def _string(raw: Mapping[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise V3StageDEvaluationError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V3StageDEvaluationError(f"{label} must be a positive integer.")
    return value


def _relative_path(value: str, base: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise V3StageDEvaluationError(f"{label} must be relative to the plan.")
    return (base / path).resolve()


def _pinned(value: object, label: str, base: Path) -> PinnedFile:
    raw = _object(value, label)
    _exact_keys(raw, {"path", "sha256"}, label)
    digest = _string(raw, "sha256", label)
    if _SHA256.fullmatch(digest) is None:
        raise V3StageDEvaluationError(f"{label}.sha256 is invalid.")
    return PinnedFile(_relative_path(_string(raw, "path", label), base, label), digest)


def _verify_pinned(pinned: PinnedFile) -> None:
    if not pinned.path.is_file() or sha256_file(pinned.path) != pinned.sha256:
        raise V3StageDEvaluationError(f"Pinned file is missing or changed: {pinned.path}")


def _labels(raw: object, label: str) -> LabelCounts:
    value = _object(raw, label)
    _exact_keys(value, {"bonafide", "spoof"}, label)
    return LabelCounts(
        bonafide=_positive_int(value.get("bonafide"), f"{label}.bonafide"),
        spoof=_positive_int(value.get("spoof"), f"{label}.spoof"),
    )


def _load_inference(raw: object) -> Inference:
    value = _object(raw, "plan.inference")
    _exact_keys(
        value,
        {
            "sample_rate",
            "window_samples",
            "batch_size",
            "num_workers",
            "device",
            "precision",
            "calibrated_probability_boundary",
            "temperature_max_iter",
        },
        "plan.inference",
    )
    workers = value.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise V3StageDEvaluationError("plan.inference.num_workers must be non-negative.")
    if (
        _string(value, "device", "plan.inference") != "cuda"
        or _string(value, "precision", "plan.inference") != "bf16"
        or value.get("calibrated_probability_boundary") != 0.5
    ):
        raise V3StageDEvaluationError("V3 inference must retain CUDA/BF16 and boundary 0.5.")
    return Inference(
        sample_rate=_positive_int(value.get("sample_rate"), "plan.inference.sample_rate"),
        window_samples=_positive_int(value.get("window_samples"), "plan.inference.window_samples"),
        batch_size=_positive_int(value.get("batch_size"), "plan.inference.batch_size"),
        num_workers=workers,
        temperature_max_iter=_positive_int(
            value.get("temperature_max_iter"), "plan.inference.temperature_max_iter"
        ),
    )


def load_plan(path: Path) -> Plan:
    """Load a plan and verify every pinned byte before model construction."""

    if not path.is_file():
        raise V3StageDEvaluationError(f"V3 Stage-D evaluation plan does not exist: {path}")
    try:
        plan_bytes = path.read_bytes()
        raw = _object(json.loads(plan_bytes), "V3 Stage-D evaluation plan")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V3StageDEvaluationError(f"Cannot read V3 Stage-D evaluation plan: {error}") from error
    _exact_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "purpose",
            "protocol",
            "license_ledger",
            "v3_governance",
            "checkpoint",
            "encoder",
            "head",
            "roles",
            "implementation",
            "inference",
            "outputs",
        },
        "V3 Stage-D evaluation plan",
    )
    if raw.get("schema_version") != SCHEMA_VERSION or _string(raw, "purpose", "plan") != "research":
        raise V3StageDEvaluationError("V3 Stage-D plan must be schema 1 and research-only.")
    run_id = _string(raw, "run_id", "plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise V3StageDEvaluationError("V3 Stage-D evaluation run_id is invalid.")
    protocol_raw = _object(raw["protocol"], "plan.protocol")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in protocol_raw.items()
    ):
        raise V3StageDEvaluationError("plan.protocol must map strings to strings.")
    protocol = cast(dict[str, str], protocol_raw)
    if protocol != _PROTOCOL:
        raise V3StageDEvaluationError("V3 Stage-D protocol limits must remain unchanged.")
    base = path.resolve().parent
    governance_raw = _object(raw["v3_governance"], "plan.v3_governance")
    _exact_keys(governance_raw, {"contract", "receipt"}, "plan.v3_governance")
    checkpoint_raw = _object(raw["checkpoint"], "plan.checkpoint")
    _exact_keys(
        checkpoint_raw,
        {"path", "sha256", "stage_b_report", "stage_b_plan", "selected_trainable_state_sha256"},
        "plan.checkpoint",
    )
    selected_state = _string(checkpoint_raw, "selected_trainable_state_sha256", "plan.checkpoint")
    if _SHA256.fullmatch(selected_state) is None:
        raise V3StageDEvaluationError("plan checkpoint selected state SHA-256 is invalid.")
    encoder_raw = _object(raw["encoder"], "plan.encoder")
    _exact_keys(encoder_raw, {"checkpoint_dir", "revision", "config", "weights"}, "plan.encoder")
    encoder_dir = _relative_path(
        _string(encoder_raw, "checkpoint_dir", "plan.encoder"), base, "encoder"
    )
    if not encoder_dir.is_dir():
        raise V3StageDEvaluationError("Pinned XLS-R encoder directory does not exist.")
    head_raw = _object(raw["head"], "plan.head")
    _exact_keys(head_raw, {"attention_size", "classifier_size", "dropout"}, "plan.head")
    dropout = head_raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise V3StageDEvaluationError("plan.head.dropout must be in [0, 1).")
    roles_raw = _object(raw["roles"], "plan.roles")
    _exact_keys(roles_raw, {"calibration", "final_ru"}, "plan.roles")
    calibration_raw = _object(roles_raw["calibration"], "plan.roles.calibration")
    _exact_keys(
        calibration_raw,
        {"manifest", "selected_split", "expected_rows", "expected_label_counts"},
        "calibration",
    )
    if _string(calibration_raw, "selected_split", "calibration") != "dev":
        raise V3StageDEvaluationError("V3 calibration role must use split='dev'.")
    final_raw = _object(roles_raw["final_ru"], "plan.roles.final_ru")
    _exact_keys(
        final_raw,
        {
            "manifest",
            "expected_rows",
            "expected_pairs",
            "expected_label_counts",
            "acoustic_gate",
            "pairing_receipt",
            "project_exposure_audit",
            "route_audit",
        },
        "final_ru",
    )
    final_labels = _labels(final_raw["expected_label_counts"], "final_ru.expected_label_counts")
    final_pairs = _positive_int(final_raw.get("expected_pairs"), "final_ru.expected_pairs")
    final_rows = _positive_int(final_raw.get("expected_rows"), "final_ru.expected_rows")
    if final_rows != 2 * final_pairs or final_labels != LabelCounts(final_pairs, final_pairs):
        raise V3StageDEvaluationError("final_ru must pin exactly balanced bonafide/spoof pairs.")
    implementation_raw = raw["implementation"]
    if not isinstance(implementation_raw, list) or not implementation_raw:
        raise V3StageDEvaluationError("plan.implementation must be a non-empty array.")
    outputs_raw = _object(raw["outputs"], "plan.outputs")
    _exact_keys(outputs_raw, {"preflight", "execution_lock", "report"}, "plan.outputs")
    outputs = Outputs(
        preflight=_relative_path(_string(outputs_raw, "preflight", "outputs"), base, "preflight"),
        execution_lock=_relative_path(
            _string(outputs_raw, "execution_lock", "outputs"), base, "execution_lock"
        ),
        report=_relative_path(_string(outputs_raw, "report", "outputs"), base, "report"),
    )
    if len({outputs.preflight, outputs.execution_lock, outputs.report}) != 3 or any(
        not item.parent.is_dir()
        for item in (outputs.preflight, outputs.execution_lock, outputs.report)
    ):
        raise V3StageDEvaluationError("V3 output paths must be distinct with existing parents.")
    plan = Plan(
        run_id=run_id,
        path=path.resolve(),
        sha256=hashlib.sha256(plan_bytes).hexdigest(),
        protocol=protocol,
        license_ledger=_pinned(raw["license_ledger"], "plan.license_ledger", base),
        governance=Governance(
            contract=_pinned(governance_raw["contract"], "plan.v3_governance.contract", base),
            receipt=_pinned(governance_raw["receipt"], "plan.v3_governance.receipt", base),
        ),
        checkpoint=V3Checkpoint(
            checkpoint=Checkpoint(
                checkpoint=_pinned(
                    {"path": checkpoint_raw["path"], "sha256": checkpoint_raw["sha256"]},
                    "plan.checkpoint",
                    base,
                ),
                stage_b_report=_pinned(
                    checkpoint_raw["stage_b_report"], "plan.checkpoint.stage_b_report", base
                ),
                selected_trainable_state_sha256=selected_state,
            ),
            stage_b_plan=_pinned(
                checkpoint_raw["stage_b_plan"], "plan.checkpoint.stage_b_plan", base
            ),
        ),
        encoder=Encoder(
            checkpoint_dir=encoder_dir,
            revision=_string(encoder_raw, "revision", "plan.encoder"),
            config=_pinned(encoder_raw["config"], "plan.encoder.config", base),
            weights=_pinned(encoder_raw["weights"], "plan.encoder.weights", base),
        ),
        head=Head(
            attention_size=_positive_int(
                head_raw.get("attention_size"), "plan.head.attention_size"
            ),
            classifier_size=_positive_int(
                head_raw.get("classifier_size"), "plan.head.classifier_size"
            ),
            dropout=float(dropout),
        ),
        calibration=CalibrationRole(
            manifest=_pinned(calibration_raw["manifest"], "calibration.manifest", base),
            expected_rows=_positive_int(calibration_raw.get("expected_rows"), "calibration.rows"),
            expected_labels=_labels(
                calibration_raw["expected_label_counts"], "calibration.expected_label_counts"
            ),
        ),
        final_ru=FinalRuRole(
            manifest=_pinned(final_raw["manifest"], "final_ru.manifest", base),
            expected_rows=final_rows,
            expected_pairs=final_pairs,
            expected_labels=final_labels,
            acoustic_gate=_pinned(final_raw["acoustic_gate"], "final_ru.acoustic_gate", base),
            pairing_receipt=_pinned(final_raw["pairing_receipt"], "final_ru.pairing_receipt", base),
            project_exposure_audit=_pinned(
                final_raw["project_exposure_audit"], "final_ru.project_exposure_audit", base
            ),
            route_audit=_pinned(final_raw["route_audit"], "final_ru.route_audit", base),
        ),
        implementation=tuple(
            _pinned(item, "implementation item", base) for item in implementation_raw
        ),
        inference=_load_inference(raw["inference"]),
        outputs=outputs,
    )
    for pinned in (
        plan.license_ledger,
        plan.governance.contract,
        plan.governance.receipt,
        plan.checkpoint.checkpoint.checkpoint,
        plan.checkpoint.checkpoint.stage_b_report,
        plan.checkpoint.stage_b_plan,
        plan.encoder.config,
        plan.encoder.weights,
        plan.calibration.manifest,
        plan.final_ru.manifest,
        plan.final_ru.acoustic_gate,
        plan.final_ru.pairing_receipt,
        plan.final_ru.project_exposure_audit,
        plan.final_ru.route_audit,
        *plan.implementation,
    ):
        _verify_pinned(pinned)
    _validate_stage_b_report(plan)
    _validate_v3_governance(plan)
    return plan


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, json.JSONDecodeError) as error:
        raise V3StageDEvaluationError(f"Cannot read {label}: {error}") from error


def _validate_stage_b_report(plan: Plan) -> None:
    report = _read_json(plan.checkpoint.checkpoint.stage_b_report.path, "Stage-B v3 report")
    run_plan = report.get("run_plan")
    if (
        report.get("status") != "ok"
        or report.get("checkpoint_scope") != "sls_head_and_final_xlsr_blocks"
        or report.get("selected_trainable_state_sha256")
        != plan.checkpoint.checkpoint.selected_trainable_state_sha256
        or report.get("frozen_final_evaluation_performed") is not False
        or report.get("calibrated") is not False
        or not isinstance(run_plan, dict)
        or run_plan.get("run_id") != "xlsr-sls-stage-b-v3"
        or run_plan.get("plan_sha256") != plan.checkpoint.stage_b_plan.sha256
    ):
        raise V3StageDEvaluationError("Pinned Stage-B v3 report does not match this contract.")


def _validate_v3_governance(plan: Plan) -> None:
    contract = _read_json(plan.governance.contract.path, "v3 governance contract")
    receipt = _read_json(plan.governance.receipt.path, "v3 governance receipt")
    controls = contract.get("controls")
    roles = contract.get("roles")
    required_controls = {
        "checkpoint_selection": "stage_a_and_stage_b_dev_loss_only",
        "calibration": "temperature_only_on_pinned_calibration_role",
        "stage_d_v2_logits_and_errors": "prohibited_for_all_v3_decisions",
        "stage_d_final_mutation": "prohibited_no_reselection_no_backfill",
        "v3_final_inference": "new_immutable_plan_one_run_after_v3_dev_selection",
    }
    if (
        contract.get("contract_id") != "xlsr-sls-v3-data-governance-v2"
        or controls != required_controls
        or not isinstance(roles, list)
        or receipt.get("status") != "validated"
        or receipt.get("contract_id") != contract.get("contract_id")
        or receipt.get("contract_sha256") != plan.governance.contract.sha256
        or receipt.get("v2_stage_d_logits_or_errors_loaded") is not False
    ):
        raise V3StageDEvaluationError(
            "Pinned v3 governance evidence is not valid for final inference."
        )
    role_by_name = {
        item.get("name"): item
        for item in roles
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for name, pinned, rows, labels in (
        (
            "calibration",
            plan.calibration.manifest,
            plan.calibration.expected_rows,
            plan.calibration.expected_labels,
        ),
        (
            "final_stage_d",
            plan.final_ru.manifest,
            plan.final_ru.expected_rows,
            plan.final_ru.expected_labels,
        ),
    ):
        role = role_by_name.get(name)
        if not isinstance(role, dict):
            raise V3StageDEvaluationError(f"V3 governance role is missing: {name}.")
        manifest = role.get("manifest")
        if (
            not isinstance(manifest, dict)
            or manifest.get("sha256") != pinned.sha256
            or role.get("expected_rows") != rows
            or role.get("expected_label_counts") != labels.as_dict()
        ):
            raise V3StageDEvaluationError(f"V3 governance role binding differs for {name}.")
    overlap = receipt.get("pairwise_overlap_counts")
    calibration_final = (
        overlap.get("calibration__final_stage_d") if isinstance(overlap, dict) else None
    )
    governance_overlap_fields = ("sample_id", "sha256", "parent_group_id", "text_hash")
    if not isinstance(calibration_final, dict) or any(
        calibration_final.get(field) != 0 for field in governance_overlap_fields
    ):
        raise V3StageDEvaluationError("V3 governance does not prove calibration/final separation.")


def _validate_final_evidence(plan: Plan, final: Sequence[ManifestRow], issues: list[str]) -> None:
    acoustic = _read_json(plan.final_ru.acoustic_gate.path, "Stage-D acoustic gate")
    assets = acoustic.get("asset_results")
    expected = {(row.sample_id, row.sha256) for row in final}
    observed = (
        {
            (item.get("sample_id"), item.get("audio_sha256"))
            for item in assets
            if isinstance(item, dict) and item.get("decision") == "pass"
        }
        if isinstance(assets, list)
        else set()
    )
    if (
        acoustic.get("all_assets_acoustically_verified") is not True
        or acoustic.get("evaluation_contract_authorized") is not True
        or acoustic.get("detector_inference_performed") is not False
        or observed != expected
    ):
        issues.append("Stage-D acoustic gate does not bind passed assets to the final manifest.")
    pairing = _read_json(plan.final_ru.pairing_receipt.path, "Stage-D pairing receipt")
    counts = pairing.get("counts")
    decision_rule = pairing.get("decision_rule")
    output_candidate = pairing.get("output_candidate")
    if (
        pairing.get("protocol_id") != "stage-d-dialogs-ru-masha-neutral-pairing-v1"
        or not isinstance(counts, dict)
        or not isinstance(decision_rule, dict)
        or not isinstance(output_candidate, dict)
        or counts.get("retained_pairs") != plan.final_ru.expected_pairs
        or decision_rule.get("detector_inference_performed") is not False
        or decision_rule.get("metric_or_detector_based_selection") is not False
        or decision_rule.get("post_selection_backfill") is not False
        or output_candidate.get("sha256") != plan.final_ru.manifest.sha256
        or output_candidate.get("rows") != plan.final_ru.expected_rows
    ):
        issues.append("Stage-D pairing receipt does not bind the frozen final pairs.")
    exposure = _read_json(
        plan.final_ru.project_exposure_audit.path, "Stage-D project exposure audit"
    )
    claims = exposure.get("claims")
    overlap = exposure.get("overlap_counts")
    if (
        not isinstance(claims, dict)
        or not isinstance(overlap, dict)
        or claims.get("exact_assets_absent_from_prior_configured_roles") is not True
        or claims.get("exact_texts_absent_from_prior_configured_roles") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("source_independent") is not False
        or claims.get("speaker_independent") is not False
        or any(overlap.get(field) != 0 for field in ("sample_id", "sha256", "text_hash"))
    ):
        issues.append("Stage-D exposure audit does not support the constrained novelty claim.")
    route = _read_json(plan.final_ru.route_audit.path, "Stage-D exact-route audit")
    claims = route.get("claims")
    gate = route.get("route_gate")
    if (
        not isinstance(claims, dict)
        or not isinstance(gate, dict)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or gate.get("exact_route_overlap_rows") != 0
    ):
        issues.append("Stage-D exact-route audit does not support the constrained novelty claim.")


def validate_inputs(plan: Plan, ledger: Mapping[str, LicenseLedgerEntry]) -> Inputs:
    calibration = tuple(
        row for row in load_manifest(plan.calibration.manifest.path) if row.split == "dev"
    )
    final = tuple(load_manifest(plan.final_ru.manifest.path))
    issues: list[str] = []
    for name, rows in (("calibration", calibration), ("final_ru", final)):
        try:
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
        except (ManifestError, V3StageDEvaluationError) as error:
            issues.extend(f"{name}: {item}" for item in error.issues)
    if len(calibration) != plan.calibration.expected_rows:
        issues.append("Calibration row count differs from the immutable plan.")
    if Counter(row.label for row in calibration) != Counter(
        plan.calibration.expected_labels.as_dict()
    ):
        issues.append("Calibration label counts differ from the immutable plan.")
    if any(row.language != "ru" or row.source_name != "pyara_ru_v7" for row in calibration):
        issues.append("Calibration role provenance changed from its locked PyAra RU contract.")
    if len(final) != plan.final_ru.expected_rows:
        issues.append("Final RU row count differs from the immutable plan.")
    if Counter(row.label for row in final) != Counter(plan.final_ru.expected_labels.as_dict()):
        issues.append("Final RU label counts differ from the immutable plan.")
    pairs: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in final:
        pairs[row.text_id].append(row)
    if len(pairs) != plan.final_ru.expected_pairs or any(
        len(pair) != 2
        or {row.label for row in pair} != {"bonafide", "spoof"}
        or len({row.text_hash for row in pair}) != 1
        for pair in pairs.values()
    ):
        issues.append("Final RU manifest does not have one exact pair per frozen text.")
    bona = [row for row in final if row.label == "bonafide"]
    spoof = [row for row in final if row.label == "spoof"]
    if any(
        row.split != "test"
        or row.language != "ru"
        or row.source_name != "common_voice_ru_v24"
        or row.code_switch != "unknown"
        for row in bona
    ) or any(
        row.split != "test"
        or row.language != "ru"
        or row.source_name != "stage_d_ru_dialogs_vits2_masha_neutral_v1"
        or row.code_switch != "false"
        for row in spoof
    ):
        issues.append("Final RU provenance changed from the frozen Stage-D pair contract.")
    for field in STAGE_B_LEAKAGE_FIELDS:
        overlap = {getattr(row, field) for row in calibration}.intersection(
            getattr(row, field) for row in final
        )
        if overlap:
            issues.append(f"Calibration/final RU leakage: {field} has {len(overlap)} overlaps.")
    _validate_final_evidence(plan, final, issues)
    if issues:
        raise V3StageDEvaluationError(issues)
    return Inputs(calibration=calibration, final_ru=final)


def _record(pinned: PinnedFile) -> dict[str, str]:
    return {"path": str(pinned.path), "sha256": pinned.sha256}


def _compute_plan(plan: Plan) -> ComputePlan:
    return ComputePlan(
        run_id=plan.run_id,
        checkpoint=plan.checkpoint.checkpoint,
        encoder=plan.encoder,
        head=plan.head,
        inference=plan.inference,
    )


def plan_record(plan: Plan) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.path),
        "plan_sha256": plan.sha256,
        "protocol": plan.protocol,
        "license_ledger": _record(plan.license_ledger),
        "v3_governance": {
            "contract": _record(plan.governance.contract),
            "receipt": _record(plan.governance.receipt),
            "v2_stage_d_logits_or_errors_loaded": False,
        },
        "checkpoint": {
            "path": str(plan.checkpoint.checkpoint.checkpoint.path),
            "sha256": plan.checkpoint.checkpoint.checkpoint.sha256,
            "stage_b_report": _record(plan.checkpoint.checkpoint.stage_b_report),
            "stage_b_plan": _record(plan.checkpoint.stage_b_plan),
            "selected_trainable_state_sha256": (
                plan.checkpoint.checkpoint.selected_trainable_state_sha256
            ),
        },
        "encoder": {
            "checkpoint_dir": str(plan.encoder.checkpoint_dir),
            "revision": plan.encoder.revision,
            "config": _record(plan.encoder.config),
            "weights": _record(plan.encoder.weights),
        },
        "head": asdict(plan.head),
        "roles": {
            "calibration": {
                "manifest": _record(plan.calibration.manifest),
                "selected_split": "dev",
                "expected_rows": plan.calibration.expected_rows,
                "expected_label_counts": plan.calibration.expected_labels.as_dict(),
            },
            "final_ru": {
                "manifest": _record(plan.final_ru.manifest),
                "expected_rows": plan.final_ru.expected_rows,
                "expected_pairs": plan.final_ru.expected_pairs,
                "expected_label_counts": plan.final_ru.expected_labels.as_dict(),
                "acoustic_gate": _record(plan.final_ru.acoustic_gate),
                "pairing_receipt": _record(plan.final_ru.pairing_receipt),
                "project_exposure_audit": _record(plan.final_ru.project_exposure_audit),
                "route_audit": _record(plan.final_ru.route_audit),
            },
        },
        "implementation": [_record(item) for item in plan.implementation],
        "inference": {
            **asdict(plan.inference),
            "device": "cuda",
            "precision": "bf16",
            "calibrated_probability_boundary": 0.5,
        },
        "outputs": {
            "preflight": str(plan.outputs.preflight),
            "execution_lock": str(plan.outputs.execution_lock),
            "report": str(plan.outputs.report),
        },
    }


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("V3 Stage-D inference requires an available CUDA BF16 device.")
    return torch.device("cuda")


def _environment(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise V3StageDEvaluationError(f"Refusing to overwrite output: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_preflight(plan: Plan) -> None:
    receipt = _read_json(plan.outputs.preflight, "v3 preflight receipt")
    run_plan = receipt.get("run_plan")
    if (
        receipt.get("status") != "validated"
        or receipt.get("mode") != "validate_only"
        or receipt.get("training_performed") is not False
        or receipt.get("detector_inference_performed") is not False
        or receipt.get("v2_stage_d_logits_or_errors_loaded") is not False
        or not isinstance(run_plan, dict)
        or run_plan.get("protocol_id") != PROTOCOL_ID
        or run_plan.get("plan_sha256") != plan.sha256
    ):
        raise V3StageDEvaluationError("V3 preflight receipt does not bind this immutable plan.")


def _preflight(plan: Plan, inputs: Inputs, device: torch.device, assets: int) -> dict[str, object]:
    return {
        "status": "validated",
        "mode": "validate_only",
        "run_plan": plan_record(plan),
        "assets_validated": assets,
        "role_rows": {"calibration": len(inputs.calibration), "final_ru": len(inputs.final_ru)},
        "environment": _environment(device),
        "training_performed": False,
        "threshold_selection_performed": False,
        "detector_inference_performed": False,
        "v2_stage_d_logits_or_errors_loaded": False,
        "pooled_language_metric": "prohibited",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    plan = load_plan(arguments.plan)
    ledger = load_license_ledger(plan.license_ledger.path)
    inputs = validate_inputs(plan, ledger)
    require_valid_assets([*inputs.calibration, *inputs.final_ru], arguments.audio_root)
    device = _cuda_device()
    preflight = _preflight(plan, inputs, device, len(inputs.calibration) + len(inputs.final_ru))
    if arguments.validate_only:
        _write_exclusive_json(plan.outputs.preflight, preflight)
        print(
            json.dumps(
                {**preflight, "preflight_sha256": sha256_file(plan.outputs.preflight)},
                ensure_ascii=False,
            )
        )
        return 0
    _require_preflight(plan)
    existing = [
        str(path) for path in (plan.outputs.execution_lock, plan.outputs.report) if path.exists()
    ]
    if existing:
        raise V3StageDEvaluationError(
            "Refusing another final v3 inference run: " + ", ".join(existing)
        )
    compute_plan = _compute_plan(plan)
    state = _load_stage_b_state(cast(Any, compute_plan))
    model = _build_model(cast(Any, compute_plan), state, device)
    execution_lock = {
        **preflight,
        "status": "calibration_and_final_ru_inference_started",
        "mode": "one_time_gpu_inference",
        "preflight": {
            "path": str(plan.outputs.preflight),
            "sha256": sha256_file(plan.outputs.preflight),
        },
        "started_at": datetime.now(UTC).isoformat(),
        "one_time_execution": True,
        "final_assets_were_previously_evaluated_by_v2": True,
        "v2_stage_d_logits_or_errors_loaded": False,
        "report_path": str(plan.outputs.report),
    }
    _write_exclusive_json(plan.outputs.execution_lock, execution_lock)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    calibration_logits, calibration_labels = _infer_logits(
        cast(Any, compute_plan), inputs.calibration, model, device, arguments.audio_root
    )
    calibration = TemperatureScaler().fit(
        calibration_logits, calibration_labels, max_iter=plan.inference.temperature_max_iter
    )
    final_logits, final_labels = _infer_logits(
        cast(Any, compute_plan), inputs.final_ru, model, device, arguments.audio_root
    )
    expected_labels = torch.tensor(
        [1.0 if row.label == "spoof" else 0.0 for row in inputs.final_ru]
    )
    if not torch.equal(final_labels, expected_labels):
        raise RuntimeError("Dataset labels differ from the frozen Stage-D manifest.")
    torch.cuda.synchronize(device)
    report = {
        **preflight,
        "status": "ok",
        "mode": "one_time_gpu_inference",
        "execution_lock": {
            "path": str(plan.outputs.execution_lock),
            "sha256": sha256_file(plan.outputs.execution_lock),
        },
        "frozen_checkpoint": {
            "path": str(plan.checkpoint.checkpoint.checkpoint.path),
            "sha256": plan.checkpoint.checkpoint.checkpoint.sha256,
            "selected_trainable_state_sha256": (
                plan.checkpoint.checkpoint.selected_trainable_state_sha256
            ),
        },
        "calibration": {
            **asdict(calibration),
            "records": len(inputs.calibration),
            "manifest": str(plan.calibration.manifest.path),
            "manifest_sha256": plan.calibration.manifest.sha256,
            "threshold_selection_performed": False,
        },
        "final_ru": _final_report(inputs.final_ru, final_logits, calibration.temperature),
        "elapsed_seconds": time.monotonic() - started,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "limitations": [
            "This is personal research, not product quality.",
            "The final Stage-D assets were previously evaluated by v2; no v2 logits or errors "
            "were loaded, and v3 checkpoint selection was locked to Stage-A/B dev loss.",
            "The layer is not source- or speaker-independent and claims only exact-route novelty, "
            "not architecture-family novelty.",
            "These metrics cannot alter this final set, select a checkpoint, threshold, "
            "architecture, or training recipe.",
        ],
    }
    _write_exclusive_json(plan.outputs.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, V3StageDEvaluationError, ValueError) as error:
        detail = (
            "\n".join(error.issues) if isinstance(error, V3StageDEvaluationError) else str(error)
        )
        print(json.dumps({"status": "error", "detail": detail}, ensure_ascii=False))
        raise SystemExit(2) from error
