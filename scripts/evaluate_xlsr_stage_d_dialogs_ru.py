#!/usr/bin/env python3
"""Preflight and execute the single locked XLS-R+SLS evaluation of the Stage-D RU layer."""

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
from typing import cast

import torch
import transformers
from torch import Tensor
from torch.torch_version import TorchVersion

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import LicenseLedgerEntry, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.stage_b_dev import STAGE_B_LEAKAGE_FIELDS
from kds.eval.calibration import TemperatureScaler, brier_score, expected_calibration_error
from kds.eval.metrics import wilson_interval
from kds.models import XlsrSlsClassifier
from kds.training import make_audio_loader
from kds.training.frozen_b0 import state_dict_sha256

SCHEMA_VERSION = 1
PROTOCOL_ID = "xlsr-sls-stage-b-v2-stage-d-dialogs-ru-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_PROTOCOL = {
    "kind": "asset_level_blind_single_ru_layer_research_evaluation",
    "quality_claim": "research_only_not_product_quality",
    "test_novelty": "exact_assets_never_inferred_project_wide_not_source_or_speaker_independent",
    "calibration": "temperature_only_on_pinned_pyara_role",
    "decision_boundary": "fixed_calibrated_probability_0.5",
    "pooled_language_metric": "prohibited",
    "final_set_model_driven_selection": "prohibited",
}


class StageDDialogsEvaluationError(ValueError):
    """Raised when an input, contract or one-time execution state is not trustworthy."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class PinnedFile:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    checkpoint: PinnedFile
    stage_b_report: PinnedFile
    selected_trainable_state_sha256: str


@dataclass(frozen=True, slots=True)
class Encoder:
    checkpoint_dir: Path
    revision: str
    config: PinnedFile
    weights: PinnedFile


@dataclass(frozen=True, slots=True)
class Head:
    attention_size: int
    classifier_size: int
    dropout: float


@dataclass(frozen=True, slots=True)
class CalibrationRole:
    manifest: PinnedFile
    expected_rows: int


@dataclass(frozen=True, slots=True)
class FinalRuRole:
    manifest: PinnedFile
    expected_rows: int
    expected_pairs: int
    acoustic_gate: PinnedFile
    project_exposure_audit: PinnedFile
    route_audit: PinnedFile


@dataclass(frozen=True, slots=True)
class Inference:
    sample_rate: int
    window_samples: int
    batch_size: int
    num_workers: int
    temperature_max_iter: int


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
    checkpoint: Checkpoint
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


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise StageDDialogsEvaluationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        raise StageDDialogsEvaluationError(
            f"{label} fields differ; missing={missing!r}, unknown={unknown!r}."
        )


def _string(raw: Mapping[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StageDDialogsEvaluationError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StageDDialogsEvaluationError(f"{label} must be a positive integer.")
    return value


def _relative_path(value: str, base: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise StageDDialogsEvaluationError(f"{label} must be relative to the plan.")
    return (base / path).resolve()


def _pinned(value: object, label: str, base: Path) -> PinnedFile:
    raw = _object(value, label)
    _exact_keys(raw, {"path", "sha256"}, label)
    digest = _string(raw, "sha256", label)
    if _SHA256.fullmatch(digest) is None:
        raise StageDDialogsEvaluationError(f"{label}.sha256 is invalid.")
    return PinnedFile(_relative_path(_string(raw, "path", label), base, label), digest)


def _verify_pinned(pinned: PinnedFile) -> None:
    if not pinned.path.is_file() or sha256_file(pinned.path) != pinned.sha256:
        raise StageDDialogsEvaluationError(f"Pinned file is missing or changed: {pinned.path}")


def load_plan(path: Path) -> Plan:
    """Load the immutable contract and verify every byte it pins before model construction."""

    if not path.is_file():
        raise StageDDialogsEvaluationError(f"Stage-D evaluation plan does not exist: {path}")
    try:
        plan_bytes = path.read_bytes()
        raw = _object(json.loads(plan_bytes), "Stage-D evaluation plan")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageDDialogsEvaluationError(
            f"Cannot read Stage-D evaluation plan: {error}"
        ) from error
    _exact_keys(
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
            "roles",
            "implementation",
            "inference",
            "outputs",
        },
        "Stage-D evaluation plan",
    )
    if raw["schema_version"] != SCHEMA_VERSION or _string(raw, "purpose", "plan") != "research":
        raise StageDDialogsEvaluationError(
            "Stage-D evaluation plan must be schema 1 and research-only."
        )
    run_id = _string(raw, "run_id", "plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise StageDDialogsEvaluationError("Stage-D evaluation run_id is invalid.")
    protocol_raw = _object(raw["protocol"], "plan.protocol")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in protocol_raw.items()
    ):
        raise StageDDialogsEvaluationError("plan.protocol must map strings to strings.")
    protocol = cast(dict[str, str], protocol_raw)
    if protocol != _PROTOCOL:
        raise StageDDialogsEvaluationError(
            "Stage-D evaluation protocol limits must remain unchanged."
        )
    base = path.resolve().parent
    checkpoint_raw = _object(raw["checkpoint"], "plan.checkpoint")
    _exact_keys(
        checkpoint_raw,
        {"path", "sha256", "stage_b_report", "selected_trainable_state_sha256"},
        "plan.checkpoint",
    )
    checkpoint = Checkpoint(
        checkpoint=_pinned(
            {"path": checkpoint_raw["path"], "sha256": checkpoint_raw["sha256"]},
            "plan.checkpoint",
            base,
        ),
        stage_b_report=_pinned(
            checkpoint_raw["stage_b_report"], "plan.checkpoint.stage_b_report", base
        ),
        selected_trainable_state_sha256=_string(
            checkpoint_raw, "selected_trainable_state_sha256", "plan.checkpoint"
        ),
    )
    if _SHA256.fullmatch(checkpoint.selected_trainable_state_sha256) is None:
        raise StageDDialogsEvaluationError("plan checkpoint selected state SHA-256 is invalid.")
    encoder_raw = _object(raw["encoder"], "plan.encoder")
    _exact_keys(encoder_raw, {"checkpoint_dir", "revision", "config", "weights"}, "plan.encoder")
    encoder_dir = _relative_path(
        _string(encoder_raw, "checkpoint_dir", "plan.encoder"), base, "encoder"
    )
    if not encoder_dir.is_dir():
        raise StageDDialogsEvaluationError("Pinned XLS-R encoder directory does not exist.")
    encoder = Encoder(
        checkpoint_dir=encoder_dir,
        revision=_string(encoder_raw, "revision", "plan.encoder"),
        config=_pinned(encoder_raw["config"], "plan.encoder.config", base),
        weights=_pinned(encoder_raw["weights"], "plan.encoder.weights", base),
    )
    head_raw = _object(raw["head"], "plan.head")
    _exact_keys(head_raw, {"attention_size", "classifier_size", "dropout"}, "plan.head")
    dropout = head_raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise StageDDialogsEvaluationError("plan.head.dropout must be in [0, 1).")
    head = Head(
        attention_size=_positive_int(head_raw.get("attention_size"), "plan.head.attention_size"),
        classifier_size=_positive_int(head_raw.get("classifier_size"), "plan.head.classifier_size"),
        dropout=float(dropout),
    )
    roles_raw = _object(raw["roles"], "plan.roles")
    _exact_keys(roles_raw, {"calibration", "final_ru"}, "plan.roles")
    calibration_raw = _object(roles_raw["calibration"], "plan.roles.calibration")
    _exact_keys(calibration_raw, {"manifest", "selected_split", "expected_rows"}, "calibration")
    if _string(calibration_raw, "selected_split", "calibration") != "dev":
        raise StageDDialogsEvaluationError("Stage-D calibration role must use split='dev'.")
    final_raw = _object(roles_raw["final_ru"], "plan.roles.final_ru")
    _exact_keys(
        final_raw,
        {
            "manifest",
            "expected_rows",
            "expected_pairs",
            "acoustic_gate",
            "project_exposure_audit",
            "route_audit",
        },
        "final_ru",
    )
    final_ru = FinalRuRole(
        manifest=_pinned(final_raw["manifest"], "final_ru.manifest", base),
        expected_rows=_positive_int(final_raw.get("expected_rows"), "final_ru.expected_rows"),
        expected_pairs=_positive_int(final_raw.get("expected_pairs"), "final_ru.expected_pairs"),
        acoustic_gate=_pinned(final_raw["acoustic_gate"], "final_ru.acoustic_gate", base),
        project_exposure_audit=_pinned(
            final_raw["project_exposure_audit"], "final_ru.project_exposure_audit", base
        ),
        route_audit=_pinned(final_raw["route_audit"], "final_ru.route_audit", base),
    )
    if final_ru.expected_rows != 2 * final_ru.expected_pairs:
        raise StageDDialogsEvaluationError("final_ru.expected_rows must be twice expected_pairs.")
    implementation_raw = raw["implementation"]
    if not isinstance(implementation_raw, list) or not implementation_raw:
        raise StageDDialogsEvaluationError("plan.implementation must be a non-empty array.")
    inference_raw = _object(raw["inference"], "plan.inference")
    _exact_keys(
        inference_raw,
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
    workers = inference_raw.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise StageDDialogsEvaluationError("plan.inference.num_workers must be non-negative.")
    if (
        _string(inference_raw, "device", "inference") != "cuda"
        or _string(inference_raw, "precision", "inference") != "bf16"
        or inference_raw.get("calibrated_probability_boundary") != 0.5
    ):
        raise StageDDialogsEvaluationError(
            "Stage-D inference must retain CUDA/BF16 and boundary 0.5."
        )
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
        raise StageDDialogsEvaluationError(
            "Stage-D output paths must be distinct with existing parents."
        )
    plan = Plan(
        run_id=run_id,
        path=path.resolve(),
        sha256=hashlib.sha256(plan_bytes).hexdigest(),
        protocol=protocol,
        license_ledger=_pinned(raw["license_ledger"], "plan.license_ledger", base),
        checkpoint=checkpoint,
        encoder=encoder,
        head=head,
        calibration=CalibrationRole(
            manifest=_pinned(calibration_raw["manifest"], "calibration.manifest", base),
            expected_rows=_positive_int(calibration_raw.get("expected_rows"), "calibration.rows"),
        ),
        final_ru=final_ru,
        implementation=tuple(
            _pinned(item, "implementation item", base) for item in implementation_raw
        ),
        inference=Inference(
            sample_rate=_positive_int(inference_raw.get("sample_rate"), "inference.sample_rate"),
            window_samples=_positive_int(
                inference_raw.get("window_samples"), "inference.window_samples"
            ),
            batch_size=_positive_int(inference_raw.get("batch_size"), "inference.batch_size"),
            num_workers=workers,
            temperature_max_iter=_positive_int(
                inference_raw.get("temperature_max_iter"), "inference.temperature_max_iter"
            ),
        ),
        outputs=outputs,
    )
    for pinned in (
        plan.license_ledger,
        plan.checkpoint.checkpoint,
        plan.checkpoint.stage_b_report,
        plan.encoder.config,
        plan.encoder.weights,
        plan.calibration.manifest,
        plan.final_ru.manifest,
        plan.final_ru.acoustic_gate,
        plan.final_ru.project_exposure_audit,
        plan.final_ru.route_audit,
        *plan.implementation,
    ):
        _verify_pinned(pinned)
    _validate_stage_b_report(plan)
    return plan


def _validate_stage_b_report(plan: Plan) -> None:
    try:
        report = _object(
            json.loads(plan.checkpoint.stage_b_report.path.read_text()), "Stage-B report"
        )
    except (OSError, json.JSONDecodeError) as error:
        raise StageDDialogsEvaluationError(f"Cannot read pinned Stage-B report: {error}") from error
    if (
        report.get("status") != "ok"
        or report.get("checkpoint_scope") != "sls_head_and_final_xlsr_blocks"
        or report.get("selected_trainable_state_sha256")
        != plan.checkpoint.selected_trainable_state_sha256
        or report.get("frozen_final_evaluation_performed") is not False
        or report.get("calibrated") is not False
    ):
        raise StageDDialogsEvaluationError("Pinned Stage-B report does not match this contract.")


def _validate_acoustic_gate(plan: Plan, final: Sequence[ManifestRow], issues: list[str]) -> None:
    try:
        report = _object(json.loads(plan.final_ru.acoustic_gate.path.read_text()), "acoustic gate")
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"Stage-D acoustic gate cannot be read: {error}")
        return
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id") != "stage-d-dialogs-ru-masha-neutral-acoustic-gate-v1"
        or report.get("all_assets_acoustically_verified") is not True
        or report.get("evaluation_contract_authorized") is not True
        or report.get("detector_inference_performed") is not False
    ):
        issues.append("Stage-D acoustic gate did not pass its required pre-inference state.")
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
    if observed != expected:
        issues.append("Stage-D acoustic pass bindings differ from the final manifest assets.")


def _validate_exposure_audit(plan: Plan, issues: list[str]) -> None:
    try:
        audit = _object(
            json.loads(plan.final_ru.project_exposure_audit.path.read_text()),
            "project exposure audit",
        )
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"Stage-D project exposure audit cannot be read: {error}")
        return
    candidate = audit.get("candidate")
    claims = audit.get("claims")
    overlaps = audit.get("overlap_counts")
    route = audit.get("route_audit")
    if (
        not isinstance(candidate, dict)
        or not isinstance(claims, dict)
        or not isinstance(overlaps, dict)
        or not isinstance(route, dict)
        or candidate.get("sha256") != plan.final_ru.manifest.sha256
        or candidate.get("rows") != plan.final_ru.expected_rows
        or route.get("sha256") != plan.final_ru.route_audit.sha256
        or claims.get("exact_assets_absent_from_prior_configured_roles") is not True
        or claims.get("exact_texts_absent_from_prior_configured_roles") is not True
        or claims.get("exact_generator_route_absent_from_prior_spoof_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("source_independent") is not False
        or claims.get("speaker_independent") is not False
        or any(overlaps.get(field) != 0 for field in ("sample_id", "sha256", "text_hash"))
    ):
        issues.append(
            "Stage-D project exposure audit does not match the frozen candidate contract."
        )


def _validate_route_audit(plan: Plan, issues: list[str]) -> None:
    try:
        audit = _object(json.loads(plan.final_ru.route_audit.path.read_text()), "exact-route audit")
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"Stage-D exact-route audit cannot be read: {error}")
        return
    claims = audit.get("claims")
    gate = audit.get("route_gate")
    if (
        audit.get("protocol_id") != "stage-d-dialogs-ru-vits2-exact-route-audit-v1"
        or not isinstance(claims, dict)
        or not isinstance(gate, dict)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("speaker_independence") is not False
        or gate.get("exact_route_overlap_rows") != 0
    ):
        issues.append("Stage-D exact-route audit does not support the claimed limited novelty.")


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
        except (ManifestError, StageDDialogsEvaluationError) as error:
            issues.extend(f"{name}: {item}" for item in error.issues)
    if len(calibration) != plan.calibration.expected_rows:
        issues.append("Calibration row count differs from the immutable plan.")
    if len(final) != plan.final_ru.expected_rows:
        issues.append("Final RU row count differs from the immutable plan.")
    labels = Counter(row.label for row in final)
    if labels != Counter(
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
        issues.append(
            "Final RU source, split, language or preserved code-switch provenance changed."
        )
    for field in STAGE_B_LEAKAGE_FIELDS:
        overlap = {getattr(row, field) for row in calibration}.intersection(
            getattr(row, field) for row in final
        )
        if overlap:
            issues.append(f"Calibration/final RU leakage: {field} has {len(overlap)} overlaps.")
    _validate_acoustic_gate(plan, final, issues)
    _validate_exposure_audit(plan, issues)
    _validate_route_audit(plan, issues)
    if issues:
        raise StageDDialogsEvaluationError(issues)
    return Inputs(calibration=calibration, final_ru=final)


def plan_record(plan: Plan) -> dict[str, object]:
    def record(item: PinnedFile) -> dict[str, str]:
        return {"path": str(item.path), "sha256": item.sha256}

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.path),
        "plan_sha256": plan.sha256,
        "protocol": plan.protocol,
        "license_ledger": record(plan.license_ledger),
        "checkpoint": {
            "path": str(plan.checkpoint.checkpoint.path),
            "sha256": plan.checkpoint.checkpoint.sha256,
            "stage_b_report": record(plan.checkpoint.stage_b_report),
            "selected_trainable_state_sha256": plan.checkpoint.selected_trainable_state_sha256,
        },
        "encoder": {
            "checkpoint_dir": str(plan.encoder.checkpoint_dir),
            "revision": plan.encoder.revision,
            "config": record(plan.encoder.config),
            "weights": record(plan.encoder.weights),
        },
        "head": asdict(plan.head),
        "roles": {
            "calibration": {
                "manifest": record(plan.calibration.manifest),
                "selected_split": "dev",
                "expected_rows": plan.calibration.expected_rows,
            },
            "final_ru": {
                "manifest": record(plan.final_ru.manifest),
                "expected_rows": plan.final_ru.expected_rows,
                "expected_pairs": plan.final_ru.expected_pairs,
                "acoustic_gate": record(plan.final_ru.acoustic_gate),
                "project_exposure_audit": record(plan.final_ru.project_exposure_audit),
                "route_audit": record(plan.final_ru.route_audit),
            },
        },
        "implementation": [record(item) for item in plan.implementation],
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
        raise RuntimeError("Stage-D XLS-R inference requires an available CUDA BF16 device.")
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
            raise StageDDialogsEvaluationError(f"Refusing to overwrite output: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_preflight(plan: Plan) -> None:
    if not plan.outputs.preflight.is_file():
        raise StageDDialogsEvaluationError(
            "The required write-once Stage-D preflight receipt is missing."
        )
    try:
        preflight = _object(json.loads(plan.outputs.preflight.read_text()), "preflight receipt")
    except (OSError, json.JSONDecodeError) as error:
        raise StageDDialogsEvaluationError(
            f"Cannot read Stage-D preflight receipt: {error}"
        ) from error
    run_plan = preflight.get("run_plan")
    if (
        preflight.get("status") != "validated"
        or preflight.get("mode") != "validate_only"
        or not isinstance(run_plan, dict)
        or run_plan.get("protocol_id") != PROTOCOL_ID
        or run_plan.get("plan_sha256") != plan.sha256
        or preflight.get("training_performed") is not False
        or preflight.get("detector_inference_performed") is not False
    ):
        raise StageDDialogsEvaluationError(
            "Stage-D preflight receipt does not bind this immutable plan."
        )


def _load_stage_b_state(plan: Plan) -> dict[str, Tensor]:
    with torch.serialization.safe_globals([TorchVersion]):
        value: object = torch.load(
            plan.checkpoint.checkpoint.path, map_location="cpu", weights_only=True
        )
    if not isinstance(value, dict):
        raise StageDDialogsEvaluationError("Frozen Stage-B checkpoint root must be a dictionary.")
    checkpoint = cast(dict[str, object], value)
    if (
        checkpoint.get("model_name") != "xlsr_sls"
        or checkpoint.get("stage") != "B"
        or checkpoint.get("training_purpose") != "research"
        or checkpoint.get("selected_trainable_state_sha256")
        != plan.checkpoint.selected_trainable_state_sha256
    ):
        raise StageDDialogsEvaluationError(
            "Frozen checkpoint does not match the planned Stage-B model."
        )
    state_value = checkpoint.get("trainable_state_dict")
    if not isinstance(state_value, dict) or not state_value:
        raise StageDDialogsEvaluationError(
            "Frozen Stage-B checkpoint has no trainable state dictionary."
        )
    state = cast(dict[str, Tensor], state_value)
    if any(
        not isinstance(key, str) or not isinstance(value, Tensor) for key, value in state.items()
    ):
        raise StageDDialogsEvaluationError("Frozen Stage-B state dictionary has invalid entries.")
    if state_dict_sha256(state) != plan.checkpoint.selected_trainable_state_sha256:
        raise StageDDialogsEvaluationError("Frozen Stage-B state digest does not match the plan.")
    allowed = ("head.", *(f"encoder.encoder.layers.{index}." for index in range(16, 24)))
    if any(not key.startswith(allowed) for key in state):
        raise StageDDialogsEvaluationError(
            "Frozen Stage-B state exceeds the declared head/tail scope."
        )
    return state


def _build_model(
    plan: Plan, state: Mapping[str, Tensor], device: torch.device
) -> XlsrSlsClassifier:
    model = XlsrSlsClassifier.from_pretrained(
        str(plan.encoder.checkpoint_dir),
        attention_size=plan.head.attention_size,
        classifier_size=plan.head.classifier_size,
        dropout=plan.head.dropout,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise StageDDialogsEvaluationError(
            f"Frozen Stage-B state has unexpected model keys: {incompatible.unexpected_keys}"
        )
    return model.eval().to(device)


def _infer_logits(
    plan: Plan,
    rows: Sequence[ManifestRow],
    model: XlsrSlsClassifier,
    device: torch.device,
    audio_root: Path,
) -> tuple[Tensor, Tensor]:
    dataset = ManifestAudioDataset(
        list(rows),
        DatasetConfig(
            audio_root=audio_root,
            sample_rate=plan.inference.sample_rate,
            window_samples=plan.inference.window_samples,
            mode="eval",
            seed=plan.run_id,
        ),
    )
    loader = make_audio_loader(
        dataset,
        batch_size=plan.inference.batch_size,
        shuffle=False,
        num_workers=plan.inference.num_workers,
        pin_memory=True,
    )
    logits_by_id: dict[str, float] = {}
    labels_by_id: dict[str, float] = {}
    with torch.inference_mode():
        for batch in loader:
            waveforms = batch.waveforms.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(waveforms)
            for sample_id, label, logit in zip(
                batch.sample_ids,
                batch.labels.tolist(),
                logits.detach().float().cpu().tolist(),
                strict=True,
            ):
                if sample_id in logits_by_id:
                    raise RuntimeError(f"Inference duplicated sample_id={sample_id!r}.")
                logits_by_id[sample_id] = float(logit)
                labels_by_id[sample_id] = float(label)
    if set(logits_by_id) != {row.sample_id for row in rows}:
        raise RuntimeError("Inference did not produce exactly one logit per manifest row.")
    return (
        torch.tensor([logits_by_id[row.sample_id] for row in rows]),
        torch.tensor([labels_by_id[row.sample_id] for row in rows]),
    )


def _metric(correct: int, examples: int) -> dict[str, object]:
    return {
        "correct": correct,
        "examples": examples,
        "value": correct / examples,
        "confidence_interval": asdict(wilson_interval(correct, examples)),
    }


def _binary_metrics(probabilities: Tensor, labels: Tensor) -> dict[str, object]:
    predictions = probabilities >= 0.5
    expected = labels >= 0.5
    bona = ~expected
    spoof = expected
    bona_correct = int((predictions[bona] == expected[bona]).sum())
    spoof_correct = int((predictions[spoof] == expected[spoof]).sum())
    return {
        "accuracy": _metric(int((predictions == expected).sum()), labels.numel()),
        "bonafide_recall": _metric(bona_correct, int(bona.sum())),
        "spoof_recall": _metric(spoof_correct, int(spoof.sum())),
        "balanced_accuracy": (bona_correct / int(bona.sum()) + spoof_correct / int(spoof.sum()))
        / 2,
        "brier_score": brier_score(probabilities, labels),
        "expected_calibration_error_15_bins": expected_calibration_error(probabilities, labels),
    }


def _final_report(
    rows: Sequence[ManifestRow], logits: Tensor, temperature: float
) -> dict[str, object]:
    labels = torch.tensor([1.0 if row.label == "spoof" else 0.0 for row in rows])
    probabilities = torch.sigmoid(logits / temperature)
    predictions = probabilities >= 0.5
    pair_indices: dict[str, list[int]] = defaultdict(list)
    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair_indices[row.text_hash].append(index)
        source_indices[row.source_name].append(index)
    both_correct = sum(
        len(indices) == 2 and bool((predictions[indices] == (labels[indices] >= 0.5)).all())
        for indices in pair_indices.values()
    )
    source_metrics = {
        source: {
            "label_counts": dict(sorted(Counter(rows[index].label for index in indices).items())),
            "class_accuracy": _metric(
                int((predictions[indices] == (labels[indices] >= 0.5)).sum()), len(indices)
            ),
            "mean_calibrated_spoof_probability": float(probabilities[indices].mean()),
        }
        for source, indices in sorted(source_indices.items())
    }
    return {
        "records": len(rows),
        "pairs": len(pair_indices),
        "metrics": _binary_metrics(probabilities, labels),
        "pairs_both_correct": _metric(both_correct, len(pair_indices)),
        "source_counts": dict(sorted(Counter(row.source_name for row in rows).items())),
        "source_metrics": source_metrics,
        "sample_results": [
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "source_name": row.source_name,
                "generator_family": row.generator_family,
                "text_hash": row.text_hash,
                "audio_sha256": row.sha256,
                "raw_logit": float(logits[index]),
                "calibrated_spoof_probability": float(probabilities[index]),
                "prediction": "spoof" if bool(predictions[index]) else "bonafide",
            }
            for index, row in enumerate(rows)
        ],
    }


def _preflight(
    plan: Plan, inputs: Inputs, device: torch.device, asset_count: int
) -> dict[str, object]:
    return {
        "status": "validated",
        "mode": "validate_only",
        "run_plan": plan_record(plan),
        "assets_validated": asset_count,
        "role_rows": {"calibration": len(inputs.calibration), "final_ru": len(inputs.final_ru)},
        "environment": _environment(device),
        "training_performed": False,
        "threshold_selection_performed": False,
        "detector_inference_performed": False,
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
        raise StageDDialogsEvaluationError(
            "Refusing another one-time Stage-D inference run because output exists: "
            + ", ".join(existing)
        )
    state = _load_stage_b_state(plan)
    model = _build_model(plan, state, device)
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
    _write_exclusive_json(plan.outputs.execution_lock, execution_lock)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    calibration_logits, calibration_labels = _infer_logits(
        plan, inputs.calibration, model, device, arguments.audio_root
    )
    calibration = TemperatureScaler().fit(
        calibration_logits, calibration_labels, max_iter=plan.inference.temperature_max_iter
    )
    final_logits, final_labels = _infer_logits(
        plan, inputs.final_ru, model, device, arguments.audio_root
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
        "final_ru": _final_report(inputs.final_ru, final_logits, calibration.temperature),
        "elapsed_seconds": time.monotonic() - started,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "limitations": [
            "This is a personal-research evaluation, not product quality.",
            "The exact Stage-D assets were absent from prior configured project roles, but the "
            "layer is not source- or speaker-independent.",
            "Only exact generator-route novelty is supported; RuASD contains generic VITS2 "
            "evidence, so architecture-family novelty is not claimed.",
            "The Common Voice base rows preserve source code_switch='unknown'; the two-review "
            "gate, not that metadata field, established Russian audibility.",
            "The metrics are a separate RU layer and must not be used to alter this final set "
            "or select a v3 model, threshold, architecture or training recipe.",
        ],
    }
    _write_exclusive_json(plan.outputs.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, StageDDialogsEvaluationError, ValueError) as error:
        detail = (
            "\n".join(error.issues)
            if isinstance(error, StageDDialogsEvaluationError)
            else str(error)
        )
        print(json.dumps({"status": "error", "detail": detail}, ensure_ascii=False))
        raise SystemExit(2) from error
