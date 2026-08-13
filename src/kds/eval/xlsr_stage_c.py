"""Immutable Stage-C evaluation contract for the frozen XLS-R+SLS v2 checkpoint.

The Stage-C suite is asset-level blind for the project, but its sources and fixed voice cannot
support source- or speaker-independent claims.  This module keeps those limitations in the
pre-inference plan and rejects a run if its exact assets, two-review gate, or exposure audit no
longer match their pinned receipts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.licenses import LicenseLedgerEntry, LicenseLedgerError, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.stage_b_dev import STAGE_B_LEAKAGE_FIELDS

STAGE_C_SCHEMA_VERSION = 1
STAGE_C_PROTOCOL_ID = "xlsr-sls-stage-b-v2-fresh-suite-stage-c-v1"
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LANGUAGES = frozenset({"ru", "kk", "mixed"})
_PROTOCOL = {
    "kind": "asset_level_blind_multilingual_research_evaluation",
    "quality_claim": "research_only_not_product_quality",
    "test_novelty": "exact_assets_never_inferred_project_wide_not_source_or_speaker_independent",
    "calibration": "temperature_only_on_pinned_pyara_role",
    "decision_boundary": "fixed_calibrated_probability_0.5",
    "pooled_language_metric": "prohibited",
}


class XlsrStageCPlanError(ValueError):
    """Raised when the Stage-C plan or a pinned input is not trustworthy."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class PinnedFile:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class FrozenCheckpoint:
    checkpoint: PinnedFile
    report: PinnedFile
    selected_trainable_state_sha256: str


@dataclass(frozen=True, slots=True)
class PinnedEncoder:
    checkpoint_dir: Path
    revision: str
    config: PinnedFile
    weights: PinnedFile


@dataclass(frozen=True, slots=True)
class FrozenHead:
    attention_size: int
    classifier_size: int
    dropout: float


@dataclass(frozen=True, slots=True)
class FrozenRole:
    manifest: PinnedFile
    selected_split: str
    expected_rows: int


@dataclass(frozen=True, slots=True)
class StageCFinalSuite:
    manifest: PinnedFile
    expected_rows: int
    expected_pairs_by_language: dict[str, int]
    full_acoustic_gate: PinnedFile
    project_exposure_audit: PinnedFile


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    sample_rate: int
    window_samples: int
    batch_size: int
    num_workers: int
    device: str
    precision: str
    calibrated_probability_boundary: float
    temperature_max_iter: int


@dataclass(frozen=True, slots=True)
class StageCOutputs:
    execution_lock: Path
    report: Path


@dataclass(frozen=True, slots=True)
class XlsrStageCPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    protocol: dict[str, str]
    license_ledger: PinnedFile
    checkpoint: FrozenCheckpoint
    encoder: PinnedEncoder
    head: FrozenHead
    calibration: FrozenRole
    final_suite: StageCFinalSuite
    implementation: tuple[PinnedFile, ...]
    inference: InferenceConfig
    outputs: StageCOutputs


@dataclass(frozen=True, slots=True)
class ValidatedStageCInputs:
    calibration: tuple[ManifestRow, ...]
    final_by_language: dict[str, tuple[ManifestRow, ...]]


def load_xlsr_stage_c_plan(path: Path) -> XlsrStageCPlan:
    """Parse a strict Stage-C plan and verify its pinned files before model construction."""

    if not path.is_file():
        raise XlsrStageCPlanError(f"Stage-C plan does not exist: {path}")
    try:
        plan_bytes = path.read_bytes()
        value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsrStageCPlanError(f"Cannot read Stage-C plan: {error}") from error
    raw = _object(value, "Stage-C plan")
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
        "Stage-C plan",
    )
    if raw["schema_version"] != STAGE_C_SCHEMA_VERSION:
        raise XlsrStageCPlanError("Stage-C plan schema_version must be 1.")
    if _string(raw, "purpose", "Stage-C plan") != "research":
        raise XlsrStageCPlanError("Stage-C plan purpose must be 'research'.")
    run_id = _string(raw, "run_id", "Stage-C plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise XlsrStageCPlanError("Stage-C run_id contains unsupported characters.")
    protocol = _string_mapping(raw["protocol"], "Stage-C protocol")
    if protocol != _PROTOCOL:
        raise XlsrStageCPlanError("Stage-C protocol limits must remain unchanged.")
    base = path.resolve().parent
    roles = _object(raw["roles"], "Stage-C roles")
    _exact_keys(roles, {"calibration", "final_suite"}, "Stage-C roles")
    plan = XlsrStageCPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        protocol=protocol,
        license_ledger=_pinned(raw["license_ledger"], "license_ledger", base),
        checkpoint=_parse_checkpoint(raw["checkpoint"], base),
        encoder=_parse_encoder(raw["encoder"], base),
        head=_parse_head(raw["head"]),
        calibration=_parse_calibration(roles["calibration"], base),
        final_suite=_parse_final_suite(roles["final_suite"], base),
        implementation=_parse_implementation(raw["implementation"], base),
        inference=_parse_inference(raw["inference"]),
        outputs=_parse_outputs(raw["outputs"], base),
    )
    pinned_files = (
        plan.license_ledger,
        plan.checkpoint.checkpoint,
        plan.checkpoint.report,
        plan.encoder.config,
        plan.encoder.weights,
        plan.calibration.manifest,
        plan.final_suite.manifest,
        plan.final_suite.full_acoustic_gate,
        plan.final_suite.project_exposure_audit,
        *plan.implementation,
    )
    for pinned in pinned_files:
        _verify_pinned(pinned)
    _validate_stage_b_report(plan)
    if plan.outputs.execution_lock == plan.outputs.report:
        raise XlsrStageCPlanError("Stage-C execution lock and report paths must differ.")
    if not plan.outputs.execution_lock.parent.is_dir() or not plan.outputs.report.parent.is_dir():
        raise XlsrStageCPlanError("Stage-C output parent directories must already exist.")
    return plan


def validate_xlsr_stage_c_inputs(
    plan: XlsrStageCPlan, ledger: Mapping[str, LicenseLedgerEntry]
) -> ValidatedStageCInputs:
    """Validate rights, exact pairs, acoustic evidence, and no-exposure disclosure."""

    calibration = _load_role(plan.calibration)
    final = tuple(load_manifest(plan.final_suite.manifest.path))
    issues: list[str] = []
    for name, rows in (("calibration", calibration), ("final_suite", final)):
        try:
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
        except (ManifestError, LicenseLedgerError) as error:
            issues.extend(f"{name}: {item}" for item in error.issues)
    try:
        validate_manifest([*calibration, *final])
    except ManifestError as error:
        issues.extend(f"Combined role validation: {item}" for item in error.issues)

    if len(final) != plan.final_suite.expected_rows:
        issues.append(
            f"Stage-C final suite expected {plan.final_suite.expected_rows} rows, got {len(final)}."
        )
    labels = Counter(row.label for row in final)
    expected_pairs = sum(plan.final_suite.expected_pairs_by_language.values())
    if labels != Counter({"bonafide": expected_pairs, "spoof": expected_pairs}):
        issues.append("Stage-C final suite is not exactly balanced by pinned pairs.")
    pair_counts = Counter((row.text_hash, row.label) for row in final)
    if len({row.text_hash for row in final}) != expected_pairs or any(
        count != 1 for count in pair_counts.values()
    ):
        issues.append(
            "Stage-C final suite does not contain exactly one bona-fide/spoof pair per text."
        )
    observed_pairs = Counter(row.language for row in final if row.label == "spoof")
    if dict(sorted(observed_pairs.items())) != plan.final_suite.expected_pairs_by_language:
        issues.append("Stage-C final suite language pair counts differ from the frozen plan.")
    if any(
        row.split != "test"
        or row.language not in _LANGUAGES
        or row.code_switch != ("true" if row.language == "mixed" else "false")
        for row in final
    ):
        issues.append("Stage-C final suite has an invalid split/language/code-switch role.")
    _validate_cross_role_overlap(calibration, final, issues)
    _validate_full_acoustic_gate(plan.final_suite, final, issues)
    _validate_project_exposure(plan.final_suite, issues)
    if issues:
        raise XlsrStageCPlanError(issues)
    return ValidatedStageCInputs(
        calibration=calibration,
        final_by_language={
            language: tuple(row for row in final if row.language == language)
            for language in ("ru", "kk", "mixed")
        },
    )


def stage_c_plan_record(plan: XlsrStageCPlan) -> dict[str, object]:
    """Return the complete frozen plan record for preflight and execution receipts."""

    return {
        "schema_version": STAGE_C_SCHEMA_VERSION,
        "protocol_id": STAGE_C_PROTOCOL_ID,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.plan_path),
        "plan_sha256": plan.plan_sha256,
        "protocol": plan.protocol,
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
        "head": {
            "attention_size": plan.head.attention_size,
            "classifier_size": plan.head.classifier_size,
            "dropout": plan.head.dropout,
        },
        "roles": {
            "calibration": {
                "manifest": _pinned_record(plan.calibration.manifest),
                "selected_split": plan.calibration.selected_split,
                "expected_rows": plan.calibration.expected_rows,
            },
            "final_suite": {
                "manifest": _pinned_record(plan.final_suite.manifest),
                "expected_rows": plan.final_suite.expected_rows,
                "expected_pairs_by_language": plan.final_suite.expected_pairs_by_language,
                "full_acoustic_gate": _pinned_record(plan.final_suite.full_acoustic_gate),
                "project_exposure_audit": _pinned_record(plan.final_suite.project_exposure_audit),
            },
        },
        "implementation": [_pinned_record(item) for item in plan.implementation],
        "inference": {
            "sample_rate": plan.inference.sample_rate,
            "window_samples": plan.inference.window_samples,
            "batch_size": plan.inference.batch_size,
            "num_workers": plan.inference.num_workers,
            "device": plan.inference.device,
            "precision": plan.inference.precision,
            "calibrated_probability_boundary": plan.inference.calibrated_probability_boundary,
            "temperature_max_iter": plan.inference.temperature_max_iter,
        },
        "outputs": {
            "execution_lock": str(plan.outputs.execution_lock),
            "report": str(plan.outputs.report),
        },
    }


def _load_role(role: FrozenRole) -> tuple[ManifestRow, ...]:
    rows = tuple(
        row for row in load_manifest(role.manifest.path) if row.split == role.selected_split
    )
    if len(rows) != role.expected_rows:
        raise XlsrStageCPlanError(
            f"Stage-C calibration expected {role.expected_rows} selected rows, got {len(rows)}."
        )
    return rows


def _validate_cross_role_overlap(
    calibration: Sequence[ManifestRow], final: Sequence[ManifestRow], issues: list[str]
) -> None:
    for field in STAGE_B_LEAKAGE_FIELDS:
        overlap = {getattr(row, field) for row in calibration}.intersection(
            getattr(row, field) for row in final
        )
        if overlap:
            issues.append(
                "Stage-C role leakage calibration/final_suite: "
                f"{field} has {len(overlap)} overlaps."
            )


def _validate_full_acoustic_gate(
    suite: StageCFinalSuite, rows: Sequence[ManifestRow], issues: list[str]
) -> None:
    try:
        value: object = json.loads(suite.full_acoustic_gate.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"Stage-C full acoustic gate cannot be read: {error}")
        return
    report = _object(value, "Stage-C full acoustic gate")
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id") != "fresh-suite-stage-c-kazakhtts-full-acoustic-gate-v1"
        or report.get("all_assets_acoustically_verified") is not True
        or report.get("immutable_inference_plan_authorized") is not True
        or report.get("detector_inference_performed") is not False
        or report.get("detector_inference_authorized") is not False
    ):
        issues.append("Stage-C full acoustic gate did not pass its immutable-plan contract.")
    results = report.get("results")
    expected = {(row.sample_id, row.sha256) for row in rows if row.label == "spoof"}
    if not isinstance(results, list):
        issues.append("Stage-C full acoustic gate lacks results.")
        return
    observed = {
        (item.get("sample_id"), item.get("audio_sha256"))
        for item in results
        if isinstance(item, dict) and item.get("decision") == "pass"
    }
    if observed != expected:
        issues.append("Stage-C acoustic pass bindings differ from the spoof manifest assets.")


def _validate_project_exposure(suite: StageCFinalSuite, issues: list[str]) -> None:
    try:
        value: object = json.loads(suite.project_exposure_audit.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"Stage-C project exposure audit cannot be read: {error}")
        return
    audit = _object(value, "Stage-C project exposure audit")
    candidate = audit.get("candidate")
    claims = audit.get("claims")
    overlaps = audit.get("overlap_counts")
    if (
        not isinstance(candidate, dict)
        or not isinstance(claims, dict)
        or not isinstance(overlaps, dict)
    ):
        issues.append("Stage-C project exposure audit schema is invalid.")
        return
    if (
        candidate.get("sha256") != suite.manifest.sha256
        or candidate.get("rows") != suite.expected_rows
        or claims.get("exact_assets_absent_from_prior_configured_roles") is not True
        or claims.get("exact_texts_absent_from_prior_configured_roles") is not True
        or claims.get("exact_generator_route_absent_from_prior_spoof_manifests") is not True
        or claims.get("source_independent") is not False
        or claims.get("speaker_independent") is not False
        or any(overlaps.get(field) != 0 for field in ("sample_id", "sha256", "text_hash"))
    ):
        issues.append(
            "Stage-C project exposure audit does not match the frozen candidate contract."
        )


def _parse_checkpoint(value: object, base: Path) -> FrozenCheckpoint:
    raw = _object(value, "checkpoint")
    _exact_keys(
        raw, {"path", "sha256", "stage_b_report", "selected_trainable_state_sha256"}, "checkpoint"
    )
    selected_hash = _string(raw, "selected_trainable_state_sha256", "checkpoint")
    if _SHA256.fullmatch(selected_hash) is None:
        raise XlsrStageCPlanError("checkpoint selected state hash is invalid.")
    return FrozenCheckpoint(
        checkpoint=_pinned(raw, "checkpoint", base, direct=True),
        report=_pinned(raw["stage_b_report"], "checkpoint.stage_b_report", base),
        selected_trainable_state_sha256=selected_hash,
    )


def _parse_encoder(value: object, base: Path) -> PinnedEncoder:
    raw = _object(value, "encoder")
    _exact_keys(raw, {"checkpoint_dir", "revision", "config", "weights"}, "encoder")
    directory = _relative_path(
        _string(raw, "checkpoint_dir", "encoder"), base, "encoder.checkpoint_dir"
    )
    if not directory.is_dir():
        raise XlsrStageCPlanError("encoder.checkpoint_dir is not a directory.")
    return PinnedEncoder(
        checkpoint_dir=directory,
        revision=_string(raw, "revision", "encoder"),
        config=_pinned(raw["config"], "encoder.config", base),
        weights=_pinned(raw["weights"], "encoder.weights", base),
    )


def _parse_head(value: object) -> FrozenHead:
    raw = _object(value, "head")
    _exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    dropout = raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise XlsrStageCPlanError("head.dropout must be in [0, 1).")
    return FrozenHead(
        attention_size=_positive_int(raw.get("attention_size"), "head.attention_size"),
        classifier_size=_positive_int(raw.get("classifier_size"), "head.classifier_size"),
        dropout=float(dropout),
    )


def _parse_calibration(value: object, base: Path) -> FrozenRole:
    raw = _object(value, "roles.calibration")
    _exact_keys(raw, {"manifest", "selected_split", "expected_rows"}, "roles.calibration")
    if _string(raw, "selected_split", "roles.calibration") != "dev":
        raise XlsrStageCPlanError("roles.calibration.selected_split must be 'dev'.")
    return FrozenRole(
        manifest=_pinned(raw["manifest"], "roles.calibration.manifest", base),
        selected_split="dev",
        expected_rows=_positive_int(raw.get("expected_rows"), "roles.calibration.expected_rows"),
    )


def _parse_final_suite(value: object, base: Path) -> StageCFinalSuite:
    raw = _object(value, "roles.final_suite")
    _exact_keys(
        raw,
        {
            "manifest",
            "expected_rows",
            "expected_pairs_by_language",
            "full_acoustic_gate",
            "project_exposure_audit",
        },
        "roles.final_suite",
    )
    counts_value = _object(raw["expected_pairs_by_language"], "expected_pairs_by_language")
    if set(counts_value) != _LANGUAGES:
        raise XlsrStageCPlanError(
            "expected_pairs_by_language must contain exactly ru, kk and mixed."
        )
    counts = {
        key: _positive_int(item, f"expected_pairs_by_language.{key}")
        for key, item in counts_value.items()
    }
    expected_rows = _positive_int(raw.get("expected_rows"), "roles.final_suite.expected_rows")
    if expected_rows != 2 * sum(counts.values()):
        raise XlsrStageCPlanError(
            "Stage-C final expected_rows must equal two times all language pairs."
        )
    return StageCFinalSuite(
        manifest=_pinned(raw["manifest"], "roles.final_suite.manifest", base),
        expected_rows=expected_rows,
        expected_pairs_by_language=dict(sorted(counts.items())),
        full_acoustic_gate=_pinned(
            raw["full_acoustic_gate"], "roles.final_suite.full_acoustic_gate", base
        ),
        project_exposure_audit=_pinned(
            raw["project_exposure_audit"], "roles.final_suite.project_exposure_audit", base
        ),
    )


def _parse_implementation(value: object, base: Path) -> tuple[PinnedFile, ...]:
    if not isinstance(value, list) or not value:
        raise XlsrStageCPlanError("implementation must be a non-empty JSON array.")
    return tuple(_pinned(item, "implementation item", base) for item in value)


def _parse_inference(value: object) -> InferenceConfig:
    raw = _object(value, "inference")
    _exact_keys(
        raw,
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
        "inference",
    )
    if (
        _string(raw, "device", "inference") != "cuda"
        or _string(raw, "precision", "inference") != "bf16"
    ):
        raise XlsrStageCPlanError("Stage-C inference requires CUDA/BF16.")
    workers = raw.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise XlsrStageCPlanError("inference.num_workers must be non-negative.")
    if raw.get("calibrated_probability_boundary") != 0.5:
        raise XlsrStageCPlanError("Calibrated probability boundary is fixed at 0.5.")
    return InferenceConfig(
        sample_rate=_positive_int(raw.get("sample_rate"), "inference.sample_rate"),
        window_samples=_positive_int(raw.get("window_samples"), "inference.window_samples"),
        batch_size=_positive_int(raw.get("batch_size"), "inference.batch_size"),
        num_workers=workers,
        device="cuda",
        precision="bf16",
        calibrated_probability_boundary=0.5,
        temperature_max_iter=_positive_int(
            raw.get("temperature_max_iter"), "inference.temperature_max_iter"
        ),
    )


def _parse_outputs(value: object, base: Path) -> StageCOutputs:
    raw = _object(value, "outputs")
    _exact_keys(raw, {"execution_lock", "report"}, "outputs")
    return StageCOutputs(
        execution_lock=_relative_path(
            _string(raw, "execution_lock", "outputs"), base, "outputs.execution_lock"
        ),
        report=_relative_path(_string(raw, "report", "outputs"), base, "outputs.report"),
    )


def _pinned(value: object, label: str, base: Path, *, direct: bool = False) -> PinnedFile:
    raw = value if direct else _object(value, label)
    if not isinstance(raw, dict):
        raise XlsrStageCPlanError(f"{label} must be an object.")
    if direct:
        path_value = _string(raw, "path", label)
        digest = _string(raw, "sha256", label)
    else:
        _exact_keys(raw, {"path", "sha256"}, label)
        path_value = _string(raw, "path", label)
        digest = _string(raw, "sha256", label)
    if _SHA256.fullmatch(digest) is None:
        raise XlsrStageCPlanError(f"{label}.sha256 is invalid.")
    return PinnedFile(_relative_path(path_value, base, f"{label}.path"), digest)


def _verify_pinned(pinned: PinnedFile) -> None:
    if not pinned.path.is_file() or sha256_file(pinned.path) != pinned.sha256:
        raise XlsrStageCPlanError(f"Pinned file is missing or changed: {pinned.path}")


def _validate_stage_b_report(plan: XlsrStageCPlan) -> None:
    try:
        value: object = json.loads(plan.checkpoint.report.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XlsrStageCPlanError(f"Cannot read Stage-B report: {error}") from error
    report = _object(value, "Stage-B report")
    if (
        report.get("status") != "ok"
        or report.get("checkpoint_scope") != "sls_head_and_final_xlsr_blocks"
        or report.get("selected_trainable_state_sha256")
        != plan.checkpoint.selected_trainable_state_sha256
        or report.get("frozen_final_evaluation_performed") is not False
        or report.get("calibrated") is not False
    ):
        raise XlsrStageCPlanError("Pinned Stage-B report does not match the Stage-C contract.")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise XlsrStageCPlanError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        raise XlsrStageCPlanError(
            f"{label} fields differ; missing={missing!r}, unknown={unknown!r}."
        )


def _string(raw: Mapping[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise XlsrStageCPlanError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _string_mapping(value: object, label: str) -> dict[str, str]:
    raw = _object(value, label)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in raw.items()):
        raise XlsrStageCPlanError(f"{label} must map strings to strings.")
    return cast(dict[str, str], raw)


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise XlsrStageCPlanError(f"{label} must be a positive integer.")
    return value


def _relative_path(value: str, base: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise XlsrStageCPlanError(f"{label} must be relative to the plan.")
    return (base / path).resolve()


def _pinned_record(pinned: PinnedFile) -> dict[str, str]:
    return {"path": str(pinned.path), "sha256": pinned.sha256}
