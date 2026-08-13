#!/usr/bin/env python3
"""Run the one locked XLS-R+SLS evaluation of the reviewed VoxForge/Qwen RU layer.

The wrapper reuses and pins the V5.5 contract validator plus the Stage-D numerical engine, while
replacing their final-layer evidence with the exact VoxForge/Qwen candidate. It never accepts a
generic manifest or mutates an earlier evaluation contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger

PROTOCOL_ID = "xlsr-sls-stage-b-v2-voxforge-ru-mdc-qwen3-tts-customvoice-aiden-v1"
_PROTOCOL = {
    "kind": "asset_level_blind_single_ru_layer_research_evaluation",
    "quality_claim": "research_only_not_product_quality",
    "test_novelty": "exact_assets_never_inferred_project_wide_not_source_or_speaker_independent",
    "calibration": "temperature_only_on_pinned_pyara_role",
    "decision_boundary": "fixed_calibrated_probability_0.5",
    "pooled_language_metric": "prohibited",
    "final_set_model_driven_selection": "prohibited",
}
_ACOUSTIC_PROTOCOL_ID = (
    "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-acoustic-gate-v1"
)
_EXPOSURE_PROTOCOL_ID = (
    "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-candidate-project-exposure-v1"
)
_ROUTE_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-exact-route-audit-v1"
_BONA_SOURCE = "voxforge_ru_mdc_2026_05"
_SPOOF_SOURCE = "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1"
_BASE: ModuleType | None = None


def _object(value: object, label: str, engine: ModuleType) -> dict[str, object]:
    return cast(dict[str, object], engine._object(value, label))


def _validate_exposure_audit(engine: ModuleType, plan: Any, issues: list[str]) -> None:
    try:
        audit = _object(
            json.loads(plan.final_ru.project_exposure_audit.path.read_text(encoding="utf-8")),
            "VoxForge/Qwen project exposure audit",
            engine,
        )
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"VoxForge/Qwen project exposure audit cannot be read: {error}")
        return
    candidate = audit.get("candidate")
    claims = audit.get("claims")
    overlaps = audit.get("overlap_counts")
    route = audit.get("route_audit")
    gate = audit.get("acoustic_gate")
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol_id") != _EXPOSURE_PROTOCOL_ID
        or audit.get("detector_inference_performed") is not False
        or audit.get("detector_inference_authorized") is not False
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
        or claims.get("russian_native_voice") is not False
        or claims.get("organizational_reviewer_independence_proven") is not False
        or any(overlaps.get(field) != 0 for field in ("sample_id", "sha256", "text_hash"))
    ):
        issues.append("VoxForge/Qwen exposure audit differs from the frozen candidate contract.")


def _validate_route_audit(engine: ModuleType, plan: Any, issues: list[str]) -> None:
    try:
        audit = _object(
            json.loads(plan.final_ru.route_audit.path.read_text(encoding="utf-8")),
            "VoxForge/Qwen exact-route audit",
            engine,
        )
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"VoxForge/Qwen exact-route audit cannot be read: {error}")
        return
    claims = audit.get("claims")
    gate = audit.get("route_gate")
    aliases = gate.get("fixed_voice_alias_overlap") if isinstance(gate, dict) else None
    aiden = aliases.get("aiden") if isinstance(aliases, dict) else None
    qualified = aliases.get("qwen3_tts_customvoice:aiden") if isinstance(aliases, dict) else None
    historical_qwen = audit.get("historical_qwen3_identifier_evidence")
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol_id") != _ROUTE_PROTOCOL_ID
        or not isinstance(claims, dict)
        or not isinstance(gate, dict)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("speaker_independence") is not False
        or claims.get("reference_audio_or_voice_cloning_used") is not False
        or gate.get("exact_route_overlap_rows") != 0
        or gate.get("generator_family_overlap_rows") != 0
        or not isinstance(aiden, dict)
        or aiden.get("rows") != 0
        or not isinstance(qualified, dict)
        or qualified.get("rows") != 0
        or not isinstance(historical_qwen, dict)
        or historical_qwen.get("rows") != 0
    ):
        issues.append("VoxForge/Qwen route audit does not support the limited exact-route claim.")


def _base() -> ModuleType:
    global _BASE
    if _BASE is not None:
        return _BASE
    path = Path(__file__).with_name("evaluate_xlsr_common_voice_ru_v24_silero_v5_5.py")
    spec = importlib.util.spec_from_file_location("kds_v5_5_evaluation_contract_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the pinned V5.5 evaluation contract base.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[spec.name]
        raise
    dynamic = cast(Any, module)
    dynamic.PROTOCOL_ID = PROTOCOL_ID
    dynamic._PROTOCOL = _PROTOCOL
    dynamic._ACOUSTIC_PROTOCOL_ID = _ACOUSTIC_PROTOCOL_ID
    dynamic._EXPOSURE_PROTOCOL_ID = _EXPOSURE_PROTOCOL_ID
    dynamic._ROUTE_PROTOCOL_ID = _ROUTE_PROTOCOL_ID
    dynamic._BONA_SOURCE = _BONA_SOURCE
    dynamic._SPOOF_SOURCE = _SPOOF_SOURCE
    dynamic._validate_exposure_audit = _validate_exposure_audit
    dynamic._validate_route_audit = _validate_route_audit
    _BASE = module
    return module


def _engine() -> ModuleType:
    return cast(ModuleType, _base()._engine())


def _validate_inputs(
    engine: ModuleType, plan: Any, ledger: Mapping[str, Any]
) -> Any:
    return _base()._validate_inputs(engine, plan, ledger)


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
            "Refusing another one-time VoxForge/Qwen inference because output exists: "
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
        calibration_logits,
        calibration_labels,
        max_iter=plan.inference.temperature_max_iter,
    )
    final_logits, final_labels = engine._infer_logits(
        plan, inputs.final_ru, model, device, arguments.audio_root
    )
    expected_labels = torch.tensor(
        [1.0 if row.label == "spoof" else 0.0 for row in inputs.final_ru]
    )
    if not torch.equal(final_labels, expected_labels):
        raise RuntimeError("Dataset labels differ from the frozen VoxForge/Qwen manifest.")
    torch.cuda.synchronize(device)
    report = {
        **preflight,
        "status": "ok",
        "mode": "one_time_gpu_inference",
        "detector_inference_performed": True,
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
        "final_ru": engine._final_report(
            inputs.final_ru, final_logits, calibration.temperature
        ),
        "elapsed_seconds": time.monotonic() - started,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "limitations": [
            "This is a personal-research evaluation, not product quality.",
            "The exact VoxForge/Qwen assets and texts were absent from prior configured roles, "
            "but the layer is not source- or speaker-independent.",
            "Only exact checkpoint/runtime-route absence is supported; this is not an "
            "architecture-family, vendor-family or Russian-native-voice claim.",
            "Distinct reviewer pseudo-IDs do not prove organizational independence.",
            "The fixed final assets, checkpoint, calibration role and 0.5 boundary must not be "
            "changed after this one-time run or tuned against its errors.",
            "VoxForge GPL-3.0-or-later and the project's personal-research limits still apply.",
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
    return _run(arguments, _engine())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LicenseLedgerError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "detail": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
