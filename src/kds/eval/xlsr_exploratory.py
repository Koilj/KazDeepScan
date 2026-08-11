"""Write-once exploratory evaluation for the frozen XLS-R Stage-B checkpoint.

This module is deliberately separate from the final/calibration path.  It permits a narrow
paired mixed stress test whose synthetic waveforms have *input-text* language provenance only.
It therefore records raw logits and the checkpoint's fixed zero-logit classifier boundary, but
never fits calibration or selects a threshold.
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
from typing import cast

from kds.data.licenses import LicenseLedgerEntry, LicenseLedgerError, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.eval.metrics import WilsonInterval, wilson_interval
from kds.training.xlsr_stage_a_plan import PinnedFile

XLSR_EXPLORATORY_MIXED_RUN_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_PROTOCOL_VALUES = {
    "kind": "exploratory_mixed_stress_test",
    "quality_claim": "not_final_quality",
    "training": "prohibited",
    "calibration": "prohibited",
    "threshold_selection": "prohibited",
    "acoustic_language_preservation": "not_performed",
}


class XlsrExploratoryMixedPlanError(ValueError):
    """Raised when the isolated mixed stress-test contract is invalid."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class PinnedXlsrEncoder:
    checkpoint_dir: Path
    revision: str
    config: PinnedFile
    weights: PinnedFile


@dataclass(frozen=True, slots=True)
class XlsrSlsHead:
    attention_size: int
    classifier_size: int
    dropout: float


@dataclass(frozen=True, slots=True)
class FrozenStageBCheckpoint:
    checkpoint: PinnedFile
    report: PinnedFile
    selected_trainable_state_sha256: str


@dataclass(frozen=True, slots=True)
class ExploratoryMixedCandidate:
    manifest: PinnedFile
    pair_lock: PinnedFile
    expected_pairs: int
    expected_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExploratoryInferenceConfig:
    sample_rate: int
    window_samples: int
    batch_size: int
    num_workers: int
    device: str
    precision: str
    raw_logit_decision_boundary: float


@dataclass(frozen=True, slots=True)
class ExploratoryOutputs:
    execution_lock: Path
    report: Path


@dataclass(frozen=True, slots=True)
class XlsrExploratoryMixedPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    protocol: dict[str, str]
    license_ledger: PinnedFile
    checkpoint: FrozenStageBCheckpoint
    encoder: PinnedXlsrEncoder
    head: XlsrSlsHead
    candidate: ExploratoryMixedCandidate
    implementation: tuple[PinnedFile, ...]
    inference: ExploratoryInferenceConfig
    outputs: ExploratoryOutputs


@dataclass(frozen=True, slots=True)
class ExploratoryMixedInputs:
    rows: tuple[ManifestRow, ...]
    pair_lock: dict[str, object]


def load_xlsr_exploratory_mixed_plan(path: Path) -> XlsrExploratoryMixedPlan:
    """Read a strict plan and verify every pinned static input before inference."""

    if not path.is_file():
        raise XlsrExploratoryMixedPlanError([f"Exploratory run plan does not exist: {path}"])
    try:
        plan_bytes = path.read_bytes()
        raw_value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsrExploratoryMixedPlanError(
            [f"Cannot read exploratory run plan {path}: {error}"]
        ) from error
    raw = _object(raw_value, "Exploratory run plan")
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
        "Exploratory run plan",
    )
    if raw["schema_version"] != XLSR_EXPLORATORY_MIXED_RUN_SCHEMA_VERSION:
        raise XlsrExploratoryMixedPlanError(
            [
                "Exploratory run plan schema_version must be "
                f"{XLSR_EXPLORATORY_MIXED_RUN_SCHEMA_VERSION!r}, got {raw['schema_version']!r}."
            ]
        )
    if _required_string(raw, "purpose", "Exploratory run plan") != "research":
        raise XlsrExploratoryMixedPlanError(
            ["Exploratory mixed plans support purpose='research' only."]
        )
    run_id = _required_string(raw, "run_id", "Exploratory run plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise XlsrExploratoryMixedPlanError(
            ["run_id must use lowercase letters, digits, dots, underscores or hyphens."]
        )

    base_directory = path.resolve().parent
    protocol = _parse_protocol(raw["protocol"])
    license_ledger = _parse_pinned_file(raw["license_ledger"], "license_ledger", base_directory)
    checkpoint = _parse_checkpoint(raw["checkpoint"], base_directory)
    encoder = _parse_encoder(raw["encoder"], base_directory)
    head = _parse_head(raw["head"])
    candidate = _parse_candidate(raw["candidate"], base_directory)
    implementation = _parse_implementation(raw["implementation"], base_directory)
    inference = _parse_inference(raw["inference"])
    outputs = _parse_outputs(raw["outputs"], base_directory)
    if outputs.execution_lock == outputs.report:
        raise XlsrExploratoryMixedPlanError(
            ["execution_lock and report output paths must be different."]
        )

    for pinned, label in (
        (license_ledger, "License ledger"),
        (checkpoint.checkpoint, "Frozen Stage-B checkpoint"),
        (checkpoint.report, "Frozen Stage-B report"),
        (encoder.config, "XLS-R config"),
        (encoder.weights, "XLS-R weights"),
        (candidate.manifest, "Mixed candidate manifest"),
        (candidate.pair_lock, "Mixed pair lock"),
        *((item, "Pinned implementation") for item in implementation),
    ):
        _verify_pinned_file(pinned, label)
    _validate_stage_b_report(checkpoint.report.path, checkpoint.selected_trainable_state_sha256)
    if not outputs.execution_lock.parent.is_dir() or not outputs.report.parent.is_dir():
        raise XlsrExploratoryMixedPlanError(
            ["Exploratory output parent directories must already exist."]
        )
    return XlsrExploratoryMixedPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        protocol=protocol,
        license_ledger=license_ledger,
        checkpoint=checkpoint,
        encoder=encoder,
        head=head,
        candidate=candidate,
        implementation=implementation,
        inference=inference,
        outputs=outputs,
    )


def validate_exploratory_mixed_inputs(
    plan: XlsrExploratoryMixedPlan, ledger: Mapping[str, LicenseLedgerEntry]
) -> ExploratoryMixedInputs:
    """Validate exactly the 30 locked bona-fide/spoof pairs and their research rights."""

    try:
        rows = load_manifest(plan.candidate.manifest.path)
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
    except (ManifestError, LicenseLedgerError) as error:
        raise XlsrExploratoryMixedPlanError(error.issues) from error
    source_ids = tuple(sorted({row.source_name for row in rows}))
    label_counts = Counter(row.label for row in rows)
    issues: list[str] = []
    expected_rows = plan.candidate.expected_pairs * 2
    if len(rows) != expected_rows:
        issues.append(
            f"Mixed candidate must contain exactly {expected_rows} rows, found {len(rows)}."
        )
    if source_ids != plan.candidate.expected_source_ids:
        issues.append(
            "Mixed candidate source IDs differ from the plan: "
            f"expected {list(plan.candidate.expected_source_ids)!r}, found {list(source_ids)!r}."
        )
    expected_label_counts = Counter(
        {"bonafide": plan.candidate.expected_pairs, "spoof": plan.candidate.expected_pairs}
    )
    if label_counts != expected_label_counts:
        issues.append(
            "Mixed candidate must contain one bonafide and one spoof row for every pinned pair."
        )
    if any(
        row.split != "test" or row.language != "mixed" or row.code_switch != "true" for row in rows
    ):
        issues.append(
            "Mixed candidate rows must all be mixed code-switch test observations."
        )
    if issues:
        raise XlsrExploratoryMixedPlanError(issues)
    pair_lock = _load_pair_lock(plan.candidate.pair_lock.path)
    _validate_pair_lock(pair_lock, rows, plan.candidate)
    return ExploratoryMixedInputs(rows=tuple(rows), pair_lock=pair_lock)


def xlsr_exploratory_mixed_plan_record(plan: XlsrExploratoryMixedPlan) -> dict[str, object]:
    """Return the complete immutable plan record suitable for an execution receipt."""

    return {
        "schema_version": XLSR_EXPLORATORY_MIXED_RUN_SCHEMA_VERSION,
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
            "pair_lock": _pinned_record(plan.candidate.pair_lock),
            "expected_pairs": plan.candidate.expected_pairs,
            "expected_source_ids": list(plan.candidate.expected_source_ids),
        },
        "implementation": [_pinned_record(item) for item in plan.implementation],
        "inference": asdict(plan.inference),
        "outputs": {
            "execution_lock": str(plan.outputs.execution_lock),
            "report": str(plan.outputs.report),
        },
    }


def pair_lock_records(pair_lock: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    """Return verified lock records in their published order for pair-level reporting."""

    pairs_value = pair_lock.get("pairs")
    if not isinstance(pairs_value, list):
        raise XlsrExploratoryMixedPlanError(["Mixed pair lock has no pairs array."])
    pairs: list[dict[str, str]] = []
    for pair in pairs_value:
        if not isinstance(pair, dict):
            raise XlsrExploratoryMixedPlanError(["Mixed pair lock contains a non-object pair."])
        parsed = {
            key: value
            for key, value in pair.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if len(parsed) != len(pair):
            raise XlsrExploratoryMixedPlanError(
                ["Mixed pair lock contains a non-string pair field."]
            )
        pairs.append(parsed)
    return tuple(pairs)


def metric_record(correct: int, examples: int) -> dict[str, object]:
    """Serialize a raw-boundary recall/accuracy metric with its finite-sample interval."""

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
    _expect_exact_keys(raw, set(_PROTOCOL_VALUES), "protocol")
    parsed = {key: _required_string(raw, key, "protocol") for key in _PROTOCOL_VALUES}
    if parsed != _PROTOCOL_VALUES:
        raise XlsrExploratoryMixedPlanError(
            [
                "Exploratory protocol must prohibit final-quality claims, training, calibration "
                "and threshold selection."
            ]
        )
    return parsed


def _parse_checkpoint(value: object, base_directory: Path) -> FrozenStageBCheckpoint:
    raw = _object(value, "checkpoint")
    _expect_exact_keys(
        raw,
        {"path", "sha256", "stage_b_report", "selected_trainable_state_sha256"},
        "checkpoint",
    )
    state_sha256 = _required_sha256(raw, "selected_trainable_state_sha256", "checkpoint")
    return FrozenStageBCheckpoint(
        checkpoint=PinnedFile(
            path=_relative_path(raw, "path", "checkpoint", base_directory),
            sha256=_required_sha256(raw, "sha256", "checkpoint"),
        ),
        report=_parse_pinned_file(raw["stage_b_report"], "stage_b_report", base_directory),
        selected_trainable_state_sha256=state_sha256,
    )


def _parse_encoder(value: object, base_directory: Path) -> PinnedXlsrEncoder:
    raw = _object(value, "encoder")
    _expect_exact_keys(raw, {"checkpoint_dir", "revision", "config", "weights"}, "encoder")
    checkpoint_dir = _relative_path(raw, "checkpoint_dir", "encoder", base_directory)
    if not checkpoint_dir.is_dir():
        raise XlsrExploratoryMixedPlanError(
            [f"Pinned XLS-R checkpoint directory does not exist: {checkpoint_dir}"]
        )
    revision = _required_string(raw, "revision", "encoder")
    if _GIT_REVISION.fullmatch(revision) is None:
        raise XlsrExploratoryMixedPlanError(
            ["encoder revision must contain a 40- or 64-character lowercase hexadecimal revision."]
        )
    return PinnedXlsrEncoder(
        checkpoint_dir=checkpoint_dir,
        revision=revision,
        config=_parse_pinned_file(raw["config"], "encoder config", base_directory),
        weights=_parse_pinned_file(raw["weights"], "encoder weights", base_directory),
    )


def _parse_head(value: object) -> XlsrSlsHead:
    raw = _object(value, "head")
    _expect_exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    return XlsrSlsHead(
        attention_size=_required_int(raw, "attention_size", "head", minimum=1),
        classifier_size=_required_int(raw, "classifier_size", "head", minimum=1),
        dropout=_required_float(raw, "dropout", "head", minimum=0.0),
    )


def _parse_candidate(value: object, base_directory: Path) -> ExploratoryMixedCandidate:
    raw = _object(value, "candidate")
    _expect_exact_keys(
        raw, {"manifest", "pair_lock", "expected_pairs", "expected_source_ids"}, "candidate"
    )
    source_ids = _string_tuple(raw["expected_source_ids"], "candidate expected_source_ids")
    if source_ids != tuple(sorted(source_ids)):
        raise XlsrExploratoryMixedPlanError(
            ["candidate expected_source_ids must be sorted for a stable plan."]
        )
    return ExploratoryMixedCandidate(
        manifest=_parse_pinned_file(raw["manifest"], "candidate manifest", base_directory),
        pair_lock=_parse_pinned_file(raw["pair_lock"], "candidate pair_lock", base_directory),
        expected_pairs=_required_int(raw, "expected_pairs", "candidate", minimum=1),
        expected_source_ids=source_ids,
    )


def _parse_implementation(value: object, base_directory: Path) -> tuple[PinnedFile, ...]:
    if not isinstance(value, list) or not value:
        raise XlsrExploratoryMixedPlanError(["implementation must be a non-empty JSON array."])
    parsed = tuple(
        _parse_pinned_file(item, f"implementation {index}", base_directory)
        for index, item in enumerate(value, start=1)
    )
    paths = [item.path for item in parsed]
    if len(paths) != len(set(paths)):
        raise XlsrExploratoryMixedPlanError(["implementation paths must not repeat."])
    return parsed


def _parse_inference(value: object) -> ExploratoryInferenceConfig:
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
    issues: list[str] = []
    if device != "cuda":
        issues.append("Exploratory XLS-R inference device must be exactly 'cuda'.")
    if precision != "bf16":
        issues.append("Exploratory XLS-R inference precision must be exactly 'bf16'.")
    if boundary != 0.0:
        issues.append(
            "Exploratory raw_logit_decision_boundary must be the fixed model default 0.0."
        )
    if issues:
        raise XlsrExploratoryMixedPlanError(issues)
    return ExploratoryInferenceConfig(
        sample_rate=_required_int(raw, "sample_rate", "inference", minimum=1),
        window_samples=_required_int(raw, "window_samples", "inference", minimum=1),
        batch_size=_required_int(raw, "batch_size", "inference", minimum=1),
        num_workers=_required_int(raw, "num_workers", "inference", minimum=0),
        device=device,
        precision=precision,
        raw_logit_decision_boundary=boundary,
    )


def _parse_outputs(value: object, base_directory: Path) -> ExploratoryOutputs:
    raw = _object(value, "outputs")
    _expect_exact_keys(raw, {"execution_lock", "report"}, "outputs")
    return ExploratoryOutputs(
        execution_lock=_relative_path(raw, "execution_lock", "outputs", base_directory),
        report=_relative_path(raw, "report", "outputs", base_directory),
    )


def _parse_pinned_file(value: object, label: str, base_directory: Path) -> PinnedFile:
    raw = _object(value, label)
    _expect_exact_keys(raw, {"path", "sha256"}, label)
    return PinnedFile(
        path=_relative_path(raw, "path", label, base_directory),
        sha256=_required_sha256(raw, "sha256", label),
    )


def _validate_stage_b_report(path: Path, state_sha256: str) -> None:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsrExploratoryMixedPlanError(
            [f"Cannot read pinned Stage-B report {path}: {error}"]
        ) from error
    report = _object(value, "Pinned Stage-B report")
    if (
        report.get("status") != "ok"
        or report.get("checkpoint_scope") != "sls_head_and_final_xlsr_blocks"
        or report.get("frozen_final_evaluation_performed") is not False
        or report.get("calibrated") is not False
        or report.get("selected_trainable_state_sha256") != state_sha256
    ):
        raise XlsrExploratoryMixedPlanError(
            ["Pinned Stage-B report is not the expected uncalibrated frozen Stage-B receipt."]
        )


def _load_pair_lock(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsrExploratoryMixedPlanError(
            [f"Cannot read mixed pair lock {path}: {error}"]
        ) from error
    return _object(value, "Mixed pair lock")


def _validate_pair_lock(
    pair_lock: Mapping[str, object], rows: list[ManifestRow], candidate: ExploratoryMixedCandidate
) -> None:
    issues: list[str] = []
    if pair_lock.get("schema_version") != 1:
        issues.append("Mixed pair lock schema_version must be 1.")
    if pair_lock.get("candidate_manifest_sha256") != candidate.manifest.sha256:
        issues.append("Mixed pair lock does not pin the candidate manifest bytes in this plan.")
    if pair_lock.get("pair_count") != candidate.expected_pairs:
        issues.append("Mixed pair lock pair_count differs from the plan.")
    try:
        pairs = pair_lock_records(pair_lock)
    except XlsrExploratoryMixedPlanError as error:
        issues.extend(error.issues)
        pairs = ()
    if len(pairs) != candidate.expected_pairs:
        issues.append("Mixed pair lock does not contain every expected pair.")
    by_text: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_text.setdefault(row.text_hash, []).append(row)
    if len(by_text) != candidate.expected_pairs:
        issues.append("Mixed candidate does not have unique text hashes for every pair.")
    expected_pairs: dict[str, tuple[ManifestRow, ManifestRow]] = {}
    for text_hash, matched in by_text.items():
        bona = [row for row in matched if row.label == "bonafide"]
        spoof = [row for row in matched if row.label == "spoof"]
        if len(matched) != 2 or len(bona) != 1 or len(spoof) != 1:
            issues.append(
                f"Mixed candidate text_hash {text_hash} is not an exact bonafide/spoof pair."
            )
            continue
        expected_pairs[text_hash] = (bona[0], spoof[0])
    locked_hashes: set[str] = set()
    required_pair_fields = {
        "annotation_id",
        "component",
        "text_hash",
        "bonafide_audio_sha256",
        "spoof_audio_sha256",
        "ru_evidence_token_indices",
        "ru_evidence_tokens",
        "kk_evidence_token_indices",
        "kk_evidence_tokens",
    }
    for number, pair in enumerate(pairs, start=1):
        if set(pair) != required_pair_fields or any(not value for value in pair.values()):
            issues.append(f"Mixed pair lock pair {number} has an invalid evidence schema.")
            continue
        text_hash = pair["text_hash"]
        if text_hash in locked_hashes:
            issues.append(f"Mixed pair lock repeats text_hash {text_hash}.")
            continue
        locked_hashes.add(text_hash)
        expected = expected_pairs.get(text_hash)
        if expected is None:
            issues.append(f"Mixed pair lock references unknown text_hash {text_hash}.")
            continue
        bona_row, spoof_row = expected
        if (
            pair["annotation_id"] != bona_row.sample_id
            or pair["bonafide_audio_sha256"] != bona_row.sha256
            or pair["spoof_audio_sha256"] != spoof_row.sha256
        ):
            issues.append(f"Mixed pair lock content mismatches candidate pair {text_hash}.")
    if locked_hashes != set(expected_pairs):
        issues.append("Mixed pair lock and candidate manifest do not name the same text pairs.")
    if issues:
        raise XlsrExploratoryMixedPlanError(issues)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise XlsrExploratoryMixedPlanError([f"{label} must be a JSON object."])
    return cast(dict[str, object], value)


def _expect_exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    issues: list[str] = []
    if missing:
        issues.append(f"{label} missing fields: " + ", ".join(missing) + ".")
    if unknown:
        issues.append(f"{label} has unknown fields: " + ", ".join(unknown) + ".")
    if issues:
        raise XlsrExploratoryMixedPlanError(issues)


def _relative_path(raw: Mapping[str, object], name: str, label: str, base_directory: Path) -> Path:
    value = _required_string(raw, name, label)
    candidate = Path(value)
    if candidate.is_absolute():
        raise XlsrExploratoryMixedPlanError([f"{label} {name} must be a relative path."])
    return (base_directory / candidate).resolve()


def _required_string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise XlsrExploratoryMixedPlanError([f"{label} {name} must be a non-empty string."])
    return value


def _required_sha256(raw: Mapping[str, object], name: str, label: str) -> str:
    value = _required_string(raw, name, label)
    if _SHA256.fullmatch(value) is None:
        raise XlsrExploratoryMixedPlanError(
            [f"{label} {name} must contain 64 lowercase hexadecimal digits."]
        )
    return value


def _required_int(raw: Mapping[str, object], name: str, label: str, *, minimum: int) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise XlsrExploratoryMixedPlanError(
            [f"{label} {name} must be an integer greater than or equal to {minimum}."]
        )
    return value


def _required_float(
    raw: Mapping[str, object], name: str, label: str, *, minimum: float = -math.inf
) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise XlsrExploratoryMixedPlanError([f"{label} {name} must be a finite number."])
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise XlsrExploratoryMixedPlanError(
            [f"{label} {name} must be a finite number greater than or equal to {minimum}."]
        )
    return number


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise XlsrExploratoryMixedPlanError([f"{label} must be a non-empty string array."])
    parsed = tuple(cast(list[str], value))
    if len(parsed) != len(set(parsed)):
        raise XlsrExploratoryMixedPlanError([f"{label} must not contain duplicates."])
    return parsed


def _verify_pinned_file(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file():
        raise XlsrExploratoryMixedPlanError([f"{label} does not exist: {pinned.path}"])
    actual = _sha256_file(pinned.path)
    if actual != pinned.sha256:
        raise XlsrExploratoryMixedPlanError(
            [f"{label} SHA-256 mismatch for {pinned.path}: expected {pinned.sha256}, got {actual}."]
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise XlsrExploratoryMixedPlanError([f"Cannot hash {path}: {error}"]) from error
    return digest.hexdigest()


def _pinned_record(pinned: PinnedFile) -> dict[str, str]:
    return {"path": str(pinned.path), "sha256": pinned.sha256}
