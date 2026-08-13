#!/usr/bin/env python3
"""Run the one locked XLS-R+SLS evaluation of the reviewed V5.5/eugene RU layer.

The numerical execution engine is imported from the already-audited Stage-D runner.  This wrapper
pins that dependency in its own plan and replaces only the immutable V5.5 candidate evidence and
provenance checks.  It never mutates the Stage-D contract or accepts a generic final layer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.stage_b_dev import STAGE_B_LEAKAGE_FIELDS

PROTOCOL_ID = "xlsr-sls-stage-b-v2-common-voice-ru-v24-silero-v5-5-eugene-v1"
_PROTOCOL = {
    "kind": "asset_level_blind_single_ru_layer_research_evaluation",
    "quality_claim": "research_only_not_product_quality",
    "test_novelty": "exact_assets_never_inferred_project_wide_not_source_or_speaker_independent",
    "calibration": "temperature_only_on_pinned_pyara_role",
    "decision_boundary": "fixed_calibrated_probability_0.5",
    "pooled_language_metric": "prohibited",
    "final_set_model_driven_selection": "prohibited",
}
_ACOUSTIC_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-acoustic-gate-v1"
_EXPOSURE_PROTOCOL_ID = (
    "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-candidate-project-exposure-v1"
)
_ROUTE_PROTOCOL_ID = "silero-v5-5-ru-eugene-exact-route-audit-v1"
_BONA_SOURCE = "common_voice_ru_v24"
_SPOOF_SOURCE = "silero_v5_5_ru_eugene_v1"


def _engine() -> ModuleType:
    path = Path(__file__).with_name("evaluate_xlsr_stage_d_dialogs_ru.py")
    spec = importlib.util.spec_from_file_location("kds_stage_d_evaluation_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned Stage-D evaluation engine.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[spec.name]
        raise
    dynamic_module = cast(Any, module)
    dynamic_module.PROTOCOL_ID = PROTOCOL_ID
    dynamic_module._PROTOCOL = _PROTOCOL
    return module


def _object(value: object, label: str, engine: ModuleType) -> dict[str, object]:
    return cast(dict[str, object], engine._object(value, label))


def _validate_acoustic_gate(
    engine: ModuleType, plan: Any, final: Sequence[ManifestRow], issues: list[str]
) -> None:
    try:
        report = _object(
            json.loads(plan.final_ru.acoustic_gate.path.read_text(encoding="utf-8")),
            "V5.5/eugene acoustic gate",
            engine,
        )
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"V5.5/eugene acoustic gate cannot be read: {error}")
        return
    items = report.get("asset_results")
    expected = {(row.sample_id, row.sha256) for row in final}
    observed = (
        {
            (item.get("sample_id"), item.get("audio_sha256"))
            for item in items
            if isinstance(item, dict) and item.get("decision") == "pass"
        }
        if isinstance(items, list)
        else set()
    )
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id") != _ACOUSTIC_PROTOCOL_ID
        or report.get("all_assets_acoustically_verified") is not True
        or report.get("evaluation_contract_authorized") is not True
        or report.get("detector_inference_performed") is not False
        or observed != expected
    ):
        issues.append("V5.5/eugene acoustic pass bindings differ from the final candidate.")


def _validate_exposure_audit(engine: ModuleType, plan: Any, issues: list[str]) -> None:
    try:
        audit = _object(
            json.loads(plan.final_ru.project_exposure_audit.path.read_text(encoding="utf-8")),
            "V5.5/eugene project exposure audit",
            engine,
        )
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"V5.5/eugene project exposure audit cannot be read: {error}")
        return
    candidate = audit.get("candidate")
    claims = audit.get("claims")
    overlaps = audit.get("overlap_counts")
    route = audit.get("route_audit")
    gate = audit.get("acoustic_gate")
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol_id") != _EXPOSURE_PROTOCOL_ID
        or not isinstance(candidate, dict)
        or not isinstance(claims, dict)
        or not isinstance(overlaps, dict)
        or not isinstance(route, dict)
        or not isinstance(gate, dict)
        or candidate.get("sha256") != plan.final_ru.manifest.sha256
        or candidate.get("rows") != plan.final_ru.expected_rows
        or candidate.get("pairs") != plan.final_ru.expected_pairs
        or route.get("sha256") != plan.final_ru.route_audit.sha256
        or gate.get("sha256") != plan.final_ru.acoustic_gate.sha256
        or claims.get("exact_assets_absent_from_prior_configured_roles") is not True
        or claims.get("exact_texts_absent_from_prior_configured_roles") is not True
        or claims.get("exact_generator_route_absent_from_prior_spoof_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("vendor_family_independent") is not False
        or claims.get("source_independent") is not False
        or claims.get("speaker_independent") is not False
        or any(overlaps.get(field) != 0 for field in ("sample_id", "sha256", "text_hash"))
    ):
        issues.append("V5.5/eugene exposure audit does not match the frozen candidate contract.")


def _validate_route_audit(engine: ModuleType, plan: Any, issues: list[str]) -> None:
    try:
        audit = _object(
            json.loads(plan.final_ru.route_audit.path.read_text(encoding="utf-8")),
            "V5.5/eugene exact-route audit",
            engine,
        )
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"V5.5/eugene exact-route audit cannot be read: {error}")
        return
    claims = audit.get("claims")
    gate = audit.get("route_gate")
    if (
        audit.get("protocol_id") != _ROUTE_PROTOCOL_ID
        or not isinstance(claims, dict)
        or not isinstance(gate, dict)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("vendor_family_independence") is not False
        or claims.get("speaker_independence") is not False
        or gate.get("exact_route_overlap_rows") != 0
    ):
        issues.append("V5.5/eugene route audit does not support the limited novelty claim.")


def _validate_inputs(engine: ModuleType, plan: Any, ledger: Mapping[str, Any]) -> Any:
    calibration = tuple(
        row for row in load_manifest(plan.calibration.manifest.path) if row.split == "dev"
    )
    final = tuple(load_manifest(plan.final_ru.manifest.path))
    issues: list[str] = []
    for name, rows in (("calibration", calibration), ("final_ru", final)):
        try:
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
        except (LicenseLedgerError, ManifestError) as error:
            issues.extend(f"{name}: {item}" for item in error.issues)
    if len(calibration) != plan.calibration.expected_rows:
        issues.append("Calibration row count differs from the immutable plan.")
    if len(final) != plan.final_ru.expected_rows:
        issues.append("Final RU row count differs from the immutable plan.")
    if Counter(row.label for row in final) != Counter(
        {"bonafide": plan.final_ru.expected_pairs, "spoof": plan.final_ru.expected_pairs}
    ):
        issues.append("Final RU manifest is not balanced by the pinned pair count.")
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
        or row.source_name != _BONA_SOURCE
        or row.code_switch != "unknown"
        for row in bona
    ) or any(
        row.split != "test"
        or row.language != "ru"
        or row.source_name != _SPOOF_SOURCE
        or row.code_switch != "unknown"
        for row in spoof
    ):
        issues.append("Final RU source, split, language or code-switch provenance changed.")
    for field in STAGE_B_LEAKAGE_FIELDS:
        overlap = {getattr(row, field) for row in calibration}.intersection(
            getattr(row, field) for row in final
        )
        if overlap:
            issues.append(f"Calibration/final RU leakage: {field} has {len(overlap)} overlaps.")
    _validate_acoustic_gate(engine, plan, final, issues)
    _validate_exposure_audit(engine, plan, issues)
    _validate_route_audit(engine, plan, issues)
    if issues:
        raise engine.StageDDialogsEvaluationError(issues)
    return engine.Inputs(calibration=calibration, final_ru=final)


def _run(arguments: argparse.Namespace, engine: ModuleType) -> int:
    plan = engine.load_plan(arguments.plan)
    ledger = load_license_ledger(plan.license_ledger.path)
    inputs = _validate_inputs(engine, plan, ledger)
    require_valid_assets([*inputs.calibration, *inputs.final_ru], arguments.audio_root)
    device = engine._cuda_device()
    preflight = engine._preflight(
        plan, inputs, device, len(inputs.calibration) + len(inputs.final_ru)
    )
    if arguments.validate_only:
        engine._write_exclusive_json(plan.outputs.preflight, preflight)
        print(json.dumps({**preflight, "preflight_sha256": sha256_file(plan.outputs.preflight)}))
        return 0
    engine._require_preflight(plan)
    existing = [
        str(path) for path in (plan.outputs.execution_lock, plan.outputs.report) if path.exists()
    ]
    if existing:
        raise engine.StageDDialogsEvaluationError(
            "Refusing another one-time V5.5/eugene inference run because output exists: "
            + ", ".join(existing)
        )
    state = engine._load_stage_b_state(plan)
    model = engine._build_model(plan, state, device)
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
        "final_assets_unseen_at_start": True,
        "report_path": str(plan.outputs.report),
    }
    engine._write_exclusive_json(plan.outputs.execution_lock, execution_lock)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    calibration_logits, calibration_labels = engine._infer_logits(
        plan, inputs.calibration, model, device, arguments.audio_root
    )
    calibration = engine.TemperatureScaler().fit(
        calibration_logits, calibration_labels, max_iter=plan.inference.temperature_max_iter
    )
    final_logits, final_labels = engine._infer_logits(
        plan, inputs.final_ru, model, device, arguments.audio_root
    )
    expected_labels = torch.tensor(
        [1.0 if row.label == "spoof" else 0.0 for row in inputs.final_ru]
    )
    if not torch.equal(final_labels, expected_labels):
        raise RuntimeError("Dataset labels differ from the frozen V5.5/eugene manifest.")
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
            "path": str(plan.checkpoint.checkpoint.path),
            "sha256": plan.checkpoint.checkpoint.sha256,
            "selected_trainable_state_sha256": plan.checkpoint.selected_trainable_state_sha256,
        },
        "calibration": {
            **asdict(calibration),
            "records": len(inputs.calibration),
            "manifest": str(plan.calibration.manifest.path),
            "manifest_sha256": plan.calibration.manifest.sha256,
            "threshold_selection_performed": False,
        },
        "final_ru": engine._final_report(inputs.final_ru, final_logits, calibration.temperature),
        "elapsed_seconds": time.monotonic() - started,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "limitations": [
            "This is a personal-research evaluation, not product quality.",
            "The exact V5.5/eugene assets were absent from prior configured roles, but the "
            "layer is not source- or speaker-independent.",
            "Only exact generator-route novelty is supported; legacy Silero evidence prevents "
            "architecture-family, vendor-family and speaker-independence claims.",
            "The Common Voice base and V5.5 spoof rows preserve code_switch='unknown'; the "
            "two-review gate, not that metadata field, established Russian audibility.",
            "The fixed final assets, checkpoint, calibration role and 0.5 boundary must not be "
            "changed after this one-time run or tuned against its errors.",
        ],
    }
    engine._write_exclusive_json(plan.outputs.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    engine = _engine()
    return _run(arguments, engine)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "detail": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
