"""Frozen calibration and confirmatory final contract for XLS-R+SLS research models.

The contract is deliberately honest about test-set history.  It can evaluate a model-version
holdout suite, but it cannot turn an asset previously inspected with an older model into a blind
project-level final.  Every such exposure is pinned and disclosed by the run plan.
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
from kds.training.xlsr_stage_a_plan import PinnedFile

XLSR_RESEARCH_FINAL_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROTOCOL = {
    "kind": "confirmatory_multilingual_research_evaluation",
    "quality_claim": "research_only_not_product_quality",
    "test_novelty": "model_version_holdout_not_project_level_blind",
    "calibration": "temperature_only_on_pinned_pyara_role",
    "decision_boundary": "fixed_calibrated_probability_0.5",
    "pooled_language_metric": "prohibited",
}
_EXPOSURE_VALUES = frozenset({"never_inferred", "previously_inferred_with_older_model"})
_EVIDENCE_VALUES = frozenset(
    {"two_review_acoustic_gate", "source_transcript_only_no_acoustic_review"}
)


class XlsrResearchFinalPlanError(ValueError):
    """Raised when a final plan or its frozen inputs cannot be trusted."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


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
    name: str
    manifest: PinnedFile
    selected_split: str
    expected_rows: int


@dataclass(frozen=True, slots=True)
class FinalLayer:
    name: str
    language: str
    manifest: PinnedFile
    expected_rows: int
    expected_pairs: int
    evidence_kind: str
    evidence_report: PinnedFile | None
    project_exposure: str
    exposure_receipt: PinnedFile | None


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
class FinalOutputs:
    execution_lock: Path
    report: Path


@dataclass(frozen=True, slots=True)
class XlsrResearchFinalPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    protocol: dict[str, str]
    license_ledger: PinnedFile
    checkpoint: FrozenCheckpoint
    encoder: PinnedEncoder
    head: FrozenHead
    train: FrozenRole
    stage_a_dev: FrozenRole
    stage_b_dev: FrozenRole
    calibration: FrozenRole
    final_layers: tuple[FinalLayer, ...]
    implementation: tuple[PinnedFile, ...]
    inference: InferenceConfig
    outputs: FinalOutputs


@dataclass(frozen=True, slots=True)
class ValidatedFinalInputs:
    train: tuple[ManifestRow, ...]
    stage_a_dev: tuple[ManifestRow, ...]
    stage_b_dev: tuple[ManifestRow, ...]
    calibration: tuple[ManifestRow, ...]
    final_layers: dict[str, tuple[ManifestRow, ...]]


def load_xlsr_research_final_plan(path: Path) -> XlsrResearchFinalPlan:
    """Parse a strict plan and verify every pinned byte before model construction."""

    if not path.is_file():
        raise XlsrResearchFinalPlanError(f"Research final plan does not exist: {path}")
    try:
        plan_bytes = path.read_bytes()
        value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsrResearchFinalPlanError(f"Cannot read research final plan: {error}") from error
    raw = _object(value, "Research final plan")
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
        "Research final plan",
    )
    if raw["schema_version"] != XLSR_RESEARCH_FINAL_SCHEMA_VERSION:
        raise XlsrResearchFinalPlanError("Research final plan schema_version must be 1.")
    if _string(raw, "purpose", "Research final plan") != "research":
        raise XlsrResearchFinalPlanError("Research final plan purpose must be 'research'.")
    run_id = _string(raw, "run_id", "Research final plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise XlsrResearchFinalPlanError("Research final run_id contains unsupported characters.")
    protocol = _string_mapping(raw["protocol"], "protocol")
    if protocol != _PROTOCOL:
        raise XlsrResearchFinalPlanError("Research final protocol limits must remain unchanged.")
    base = path.resolve().parent
    roles = _object(raw["roles"], "roles")
    _exact_keys(
        roles,
        {"train", "stage_a_dev", "stage_b_dev", "calibration", "final_layers"},
        "roles",
    )
    layers_value = roles["final_layers"]
    if not isinstance(layers_value, list):
        raise XlsrResearchFinalPlanError("roles.final_layers must be a JSON array.")
    layers = tuple(_parse_final_layer(item, base) for item in layers_value)
    if tuple(layer.name for layer in layers) != ("ru", "kk", "mixed"):
        raise XlsrResearchFinalPlanError("Final layers must be ordered exactly as ru, kk, mixed.")
    plan = XlsrResearchFinalPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        protocol=protocol,
        license_ledger=_pinned(raw["license_ledger"], "license_ledger", base),
        checkpoint=_parse_checkpoint(raw["checkpoint"], base),
        encoder=_parse_encoder(raw["encoder"], base),
        head=_parse_head(raw["head"]),
        train=_parse_role(roles["train"], "train", base),
        stage_a_dev=_parse_role(roles["stage_a_dev"], "stage_a_dev", base),
        stage_b_dev=_parse_role(roles["stage_b_dev"], "stage_b_dev", base),
        calibration=_parse_role(roles["calibration"], "calibration", base),
        final_layers=layers,
        implementation=_parse_implementation(raw["implementation"], base),
        inference=_parse_inference(raw["inference"]),
        outputs=_parse_outputs(raw["outputs"], base),
    )
    pinned_files = [
        plan.license_ledger,
        plan.checkpoint.checkpoint,
        plan.checkpoint.report,
        plan.encoder.config,
        plan.encoder.weights,
        plan.train.manifest,
        plan.stage_a_dev.manifest,
        plan.stage_b_dev.manifest,
        plan.calibration.manifest,
        *(layer.manifest for layer in plan.final_layers),
        *(layer.evidence_report for layer in plan.final_layers if layer.evidence_report),
        *(layer.exposure_receipt for layer in plan.final_layers if layer.exposure_receipt),
        *plan.implementation,
    ]
    for pinned in pinned_files:
        _verify_pinned(pinned)
    _validate_stage_b_report(plan)
    if plan.outputs.execution_lock == plan.outputs.report:
        raise XlsrResearchFinalPlanError("Final execution lock and report paths must differ.")
    if not plan.outputs.execution_lock.parent.is_dir() or not plan.outputs.report.parent.is_dir():
        raise XlsrResearchFinalPlanError("Final output parent directories must already exist.")
    return plan


def validate_xlsr_research_final_inputs(
    plan: XlsrResearchFinalPlan, ledger: Mapping[str, LicenseLedgerEntry]
) -> ValidatedFinalInputs:
    """Validate role shape, rights, leakage, generator novelty and evidence disclosures."""

    train = _load_role(plan.train)
    stage_a_dev = _load_role(plan.stage_a_dev)
    stage_b_dev = _load_role(plan.stage_b_dev)
    calibration = _load_role(plan.calibration)
    final_layers = {
        layer.name: tuple(load_manifest(layer.manifest.path)) for layer in plan.final_layers
    }
    issues: list[str] = []
    all_roles = {
        "train": train,
        "stage_a_dev": stage_a_dev,
        "stage_b_dev": stage_b_dev,
        "calibration": calibration,
        **final_layers,
    }
    for name, rows in all_roles.items():
        try:
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
        except (ManifestError, LicenseLedgerError) as error:
            issues.extend(f"{name}: {item}" for item in error.issues)
    observed = (*train, *stage_a_dev, *stage_b_dev, *calibration)
    try:
        validate_manifest([*observed, *(row for rows in final_layers.values() for row in rows)])
    except ManifestError as error:
        issues.extend(f"Combined role validation: {item}" for item in error.issues)
    observed_sources = {row.source_name for row in observed}
    observed_spoof_families = {row.generator_family for row in observed if row.label == "spoof"}
    for layer in plan.final_layers:
        rows = final_layers[layer.name]
        labels = Counter(row.label for row in rows)
        if len(rows) != layer.expected_rows:
            issues.append(
                f"Final layer {layer.name} expected {layer.expected_rows} rows, got {len(rows)}."
            )
        if labels != Counter({"bonafide": layer.expected_pairs, "spoof": layer.expected_pairs}):
            issues.append(f"Final layer {layer.name} is not exactly balanced by pinned pairs.")
        if len({row.text_hash for row in rows}) != layer.expected_pairs:
            issues.append(f"Final layer {layer.name} does not have one text group per pair.")
        pair_counts = Counter((row.text_hash, row.label) for row in rows)
        if any(count != 1 for count in pair_counts.values()):
            issues.append(f"Final layer {layer.name} contains duplicate text/label bindings.")
        expected_switch = "true" if layer.language == "mixed" else "false"
        if any(
            row.split != "test"
            or row.language != layer.language
            or row.code_switch != expected_switch
            for row in rows
        ):
            issues.append(
                f"Final layer {layer.name} has an invalid split/language/code-switch role."
            )
        final_sources = {row.source_name for row in rows}
        overlap_sources = sorted(final_sources.intersection(observed_sources))
        if overlap_sources:
            sources = ", ".join(overlap_sources)
            issues.append(
                f"Final layer {layer.name} reuses observed source IDs: {sources}."
            )
        final_families = {row.generator_family for row in rows if row.label == "spoof"}
        overlap_families = sorted(final_families.intersection(observed_spoof_families))
        if overlap_families:
            issues.append(
                f"Final layer {layer.name} reuses observed spoof families: "
                + ", ".join(overlap_families)
                + "."
            )
        _validate_evidence(layer, rows, issues)
    _validate_pairwise_role_overlap(all_roles, issues)
    if issues:
        raise XlsrResearchFinalPlanError(issues)
    return ValidatedFinalInputs(
        train=train,
        stage_a_dev=stage_a_dev,
        stage_b_dev=stage_b_dev,
        calibration=calibration,
        final_layers=final_layers,
    )


def final_plan_record(plan: XlsrResearchFinalPlan) -> dict[str, object]:
    """Return the immutable plan receipt included in preflight and final reports."""

    return {
        "schema_version": XLSR_RESEARCH_FINAL_SCHEMA_VERSION,
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
            "train": _role_record(plan.train),
            "stage_a_dev": _role_record(plan.stage_a_dev),
            "stage_b_dev": _role_record(plan.stage_b_dev),
            "calibration": _role_record(plan.calibration),
            "final_layers": [_layer_record(layer) for layer in plan.final_layers],
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
        raise XlsrResearchFinalPlanError(
            f"Role {role.name} expected {role.expected_rows} selected rows, got {len(rows)}."
        )
    return rows


def _validate_pairwise_role_overlap(
    roles: Mapping[str, Sequence[ManifestRow]], issues: list[str]
) -> None:
    names = tuple(roles)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = roles[left_name]
            right = roles[right_name]
            for field in STAGE_B_LEAKAGE_FIELDS:
                left_values = {getattr(row, field) for row in left}
                overlap = left_values.intersection(getattr(row, field) for row in right)
                if overlap:
                    overlap_count = len(overlap)
                    issues.append(
                        f"Role leakage {left_name}/{right_name}: "
                        f"{field} has {overlap_count} overlaps."
                    )


def _validate_evidence(layer: FinalLayer, rows: Sequence[ManifestRow], issues: list[str]) -> None:
    if layer.evidence_kind == "two_review_acoustic_gate":
        if layer.evidence_report is None:
            issues.append(f"Final layer {layer.name} lacks its acoustic gate report.")
            return
        try:
            value: object = json.loads(layer.evidence_report.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"Final layer {layer.name} acoustic report cannot be read: {error}")
            return
        report = _object(value, f"Final layer {layer.name} acoustic report")
        if report.get("all_assets_acoustically_verified") is not True:
            issues.append(f"Final layer {layer.name} did not pass every acoustic asset review.")
        results = report.get("asset_results")
        expected = {(row.sample_id, row.sha256) for row in rows}
        if not isinstance(results, list):
            issues.append(f"Final layer {layer.name} acoustic report lacks asset_results.")
            return
        observed = {
            (item.get("sample_id"), item.get("audio_sha256"))
            for item in results
            if isinstance(item, dict) and item.get("decision") == "pass"
        }
        if observed != expected:
            issues.append(f"Final layer {layer.name} acoustic pass bindings differ from manifest.")
    elif layer.evidence_report is not None:
        issues.append(
            f"Final layer {layer.name} source-transcript evidence cannot pin a gate report."
        )
    if layer.project_exposure == "previously_inferred_with_older_model":
        if layer.exposure_receipt is None:
            issues.append(f"Final layer {layer.name} must pin its prior-inference receipt.")
    elif layer.exposure_receipt is not None:
        issues.append(
            f"Final layer {layer.name} declares no inference but pins an exposure receipt."
        )


def _parse_checkpoint(value: object, base: Path) -> FrozenCheckpoint:
    raw = _object(value, "checkpoint")
    _exact_keys(
        raw, {"path", "sha256", "stage_b_report", "selected_trainable_state_sha256"}, "checkpoint"
    )
    state_hash = _string(raw, "selected_trainable_state_sha256", "checkpoint")
    if _SHA256.fullmatch(state_hash) is None:
        raise XlsrResearchFinalPlanError("checkpoint selected state hash is invalid.")
    return FrozenCheckpoint(
        checkpoint=_pinned(raw, "checkpoint", base, direct=True),
        report=_pinned(raw["stage_b_report"], "checkpoint.stage_b_report", base),
        selected_trainable_state_sha256=state_hash,
    )


def _parse_encoder(value: object, base: Path) -> PinnedEncoder:
    raw = _object(value, "encoder")
    _exact_keys(raw, {"checkpoint_dir", "revision", "config", "weights"}, "encoder")
    directory = _relative_path(
        _string(raw, "checkpoint_dir", "encoder"), base, "encoder.checkpoint_dir"
    )
    if not directory.is_dir():
        raise XlsrResearchFinalPlanError("encoder.checkpoint_dir is not a directory.")
    return PinnedEncoder(
        checkpoint_dir=directory,
        revision=_string(raw, "revision", "encoder"),
        config=_pinned(raw["config"], "encoder.config", base),
        weights=_pinned(raw["weights"], "encoder.weights", base),
    )


def _parse_head(value: object) -> FrozenHead:
    raw = _object(value, "head")
    _exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    attention = _positive_int(raw.get("attention_size"), "head.attention_size")
    classifier = _positive_int(raw.get("classifier_size"), "head.classifier_size")
    dropout = raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise XlsrResearchFinalPlanError("head.dropout must be in [0, 1).")
    return FrozenHead(attention, classifier, float(dropout))


def _parse_role(value: object, name: str, base: Path) -> FrozenRole:
    raw = _object(value, f"roles.{name}")
    _exact_keys(raw, {"manifest", "selected_split", "expected_rows"}, f"roles.{name}")
    split = _string(raw, "selected_split", f"roles.{name}")
    expected_split = "train" if name == "train" else "dev"
    if split != expected_split:
        raise XlsrResearchFinalPlanError(f"roles.{name}.selected_split must be {expected_split!r}.")
    return FrozenRole(
        name=name,
        manifest=_pinned(raw["manifest"], f"roles.{name}.manifest", base),
        selected_split=split,
        expected_rows=_positive_int(raw.get("expected_rows"), f"roles.{name}.expected_rows"),
    )


def _parse_final_layer(value: object, base: Path) -> FinalLayer:
    raw = _object(value, "final layer")
    _exact_keys(
        raw,
        {
            "name",
            "language",
            "manifest",
            "expected_rows",
            "expected_pairs",
            "evidence_kind",
            "evidence_report",
            "project_exposure",
            "exposure_receipt",
        },
        "final layer",
    )
    name = _string(raw, "name", "final layer")
    language = _string(raw, "language", "final layer")
    if name not in {"ru", "kk", "mixed"} or language != name:
        raise XlsrResearchFinalPlanError(
            "Final layer name/language must be matching ru, kk or mixed."
        )
    evidence_kind = _string(raw, "evidence_kind", "final layer")
    exposure = _string(raw, "project_exposure", "final layer")
    if evidence_kind not in _EVIDENCE_VALUES:
        raise XlsrResearchFinalPlanError("Final layer evidence_kind is unsupported.")
    if exposure not in _EXPOSURE_VALUES:
        raise XlsrResearchFinalPlanError("Final layer project_exposure is unsupported.")
    evidence_value = raw["evidence_report"]
    exposure_value = raw["exposure_receipt"]
    evidence = None if evidence_value is None else _pinned(evidence_value, "evidence_report", base)
    receipt = None if exposure_value is None else _pinned(exposure_value, "exposure_receipt", base)
    pairs = _positive_int(raw.get("expected_pairs"), "final layer expected_pairs")
    rows = _positive_int(raw.get("expected_rows"), "final layer expected_rows")
    if rows != pairs * 2:
        raise XlsrResearchFinalPlanError(
            "Final layer expected_rows must equal two times expected_pairs."
        )
    return FinalLayer(
        name,
        language,
        _pinned(raw["manifest"], "final layer manifest", base),
        rows,
        pairs,
        evidence_kind,
        evidence,
        exposure,
        receipt,
    )


def _parse_implementation(value: object, base: Path) -> tuple[PinnedFile, ...]:
    if not isinstance(value, list) or not value:
        raise XlsrResearchFinalPlanError("implementation must be a non-empty JSON array.")
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
        raise XlsrResearchFinalPlanError("Research final inference requires CUDA/BF16.")
    workers = raw.get("num_workers")
    boundary = raw.get("calibrated_probability_boundary")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise XlsrResearchFinalPlanError("inference.num_workers must be non-negative.")
    if boundary != 0.5:
        raise XlsrResearchFinalPlanError("Calibrated probability boundary is fixed at 0.5.")
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


def _parse_outputs(value: object, base: Path) -> FinalOutputs:
    raw = _object(value, "outputs")
    _exact_keys(raw, {"execution_lock", "report"}, "outputs")
    return FinalOutputs(
        _relative_path(_string(raw, "execution_lock", "outputs"), base, "outputs.execution_lock"),
        _relative_path(_string(raw, "report", "outputs"), base, "outputs.report"),
    )


def _pinned(value: object, label: str, base: Path, *, direct: bool = False) -> PinnedFile:
    raw = value if direct else _object(value, label)
    if not isinstance(raw, dict):
        raise XlsrResearchFinalPlanError(f"{label} must be an object.")
    if direct:
        path_value = _string(raw, "path", label)
        digest = _string(raw, "sha256", label)
    else:
        _exact_keys(raw, {"path", "sha256"}, label)
        path_value = _string(raw, "path", label)
        digest = _string(raw, "sha256", label)
    if _SHA256.fullmatch(digest) is None:
        raise XlsrResearchFinalPlanError(f"{label}.sha256 is invalid.")
    return PinnedFile(_relative_path(path_value, base, f"{label}.path"), digest)


def _verify_pinned(pinned: PinnedFile) -> None:
    if not pinned.path.is_file() or sha256_file(pinned.path) != pinned.sha256:
        raise XlsrResearchFinalPlanError(f"Pinned file is missing or changed: {pinned.path}")


def _validate_stage_b_report(plan: XlsrResearchFinalPlan) -> None:
    try:
        value: object = json.loads(plan.checkpoint.report.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XlsrResearchFinalPlanError(f"Cannot read Stage-B report: {error}") from error
    report = _object(value, "Stage-B report")
    if (
        report.get("status") != "ok"
        or report.get("checkpoint_scope") != "sls_head_and_final_xlsr_blocks"
        or report.get("selected_trainable_state_sha256")
        != plan.checkpoint.selected_trainable_state_sha256
        or report.get("frozen_final_evaluation_performed") is not False
        or report.get("calibrated") is not False
    ):
        raise XlsrResearchFinalPlanError("Pinned Stage-B report does not match the final contract.")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise XlsrResearchFinalPlanError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        raise XlsrResearchFinalPlanError(
            f"{label} fields differ; missing={missing!r}, unknown={unknown!r}."
        )


def _string(raw: Mapping[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise XlsrResearchFinalPlanError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _string_mapping(value: object, label: str) -> dict[str, str]:
    raw = _object(value, label)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in raw.items()):
        raise XlsrResearchFinalPlanError(f"{label} must map strings to strings.")
    return cast(dict[str, str], raw)


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise XlsrResearchFinalPlanError(f"{label} must be a positive integer.")
    return value


def _relative_path(value: str, base: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise XlsrResearchFinalPlanError(f"{label} must be relative to the plan.")
    return (base / path).resolve()


def _pinned_record(pinned: PinnedFile) -> dict[str, str]:
    return {"path": str(pinned.path), "sha256": pinned.sha256}


def _role_record(role: FrozenRole) -> dict[str, object]:
    return {
        "manifest": _pinned_record(role.manifest),
        "selected_split": role.selected_split,
        "expected_rows": role.expected_rows,
    }


def _layer_record(layer: FinalLayer) -> dict[str, object]:
    return {
        "name": layer.name,
        "language": layer.language,
        "manifest": _pinned_record(layer.manifest),
        "expected_rows": layer.expected_rows,
        "expected_pairs": layer.expected_pairs,
        "evidence_kind": layer.evidence_kind,
        "evidence_report": _pinned_record(layer.evidence_report) if layer.evidence_report else None,
        "project_exposure": layer.project_exposure,
        "exposure_receipt": _pinned_record(layer.exposure_receipt)
        if layer.exposure_receipt
        else None,
    }
