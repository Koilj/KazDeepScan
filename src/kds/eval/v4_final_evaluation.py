"""Fail-closed one-time final evaluation for the reviewed XLS-R+SLS v4 pairs.

This module deliberately has no route for fitting a parameter, choosing a threshold, or
publishing a pooled RU+KK headline.  It can only score the immutable reconciliation pair lock
once after a no-logit preflight has revalidated every final asset and role boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import transformers
from torch import Tensor
from torch.torch_version import TorchVersion

from kds.data.assets import sha256_file
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import LicenseLedgerEntry, LicenseLedgerError, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.eval.calibration import brier_score, expected_calibration_error
from kds.eval.metrics import wilson_interval
from kds.models import XlsrSlsClassifier
from kds.training import make_audio_loader
from kds.training.frozen_b0 import state_dict_sha256

SCHEMA_VERSION = 1
PROTOCOL_ID = "xlsr-sls-model-v4-final-reconciliation-evaluation-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_PROTOCOL = {
    "kind": "one_time_final_reconciliation_evaluation",
    "purpose": "personal_research_only",
    "final_inference": "authorized_exactly_once_after_no_logit_preflight",
    "calibration": "reuse_fixed_ru_temperature_only",
    "kk_probability_claim": "prohibited",
    "threshold_selection": "prohibited",
    "pair_mutation_or_backfill": "prohibited",
    "training_or_checkpoint_mutation": "prohibited",
    "output_overwrite": "prohibited",
    "detector_feedback": "prohibited",
    "network_downloads": "prohibited",
    "pooled_ru_kk_headline": "prohibited",
    "re_evaluation": "prohibited",
}


class V4FinalEvaluationError(ValueError):
    """Raised when a final-evaluation contract cannot be proven safe to run."""

    def __init__(self, issues: Iterable[str] | str) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class PinnedFile:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    path: Path
    sha256: str
    selected_state_sha256: str


@dataclass(frozen=True, slots=True)
class Encoder:
    directory: Path
    revision: str
    config: PinnedFile
    weights: PinnedFile


@dataclass(frozen=True, slots=True)
class Head:
    attention_size: int
    classifier_size: int
    dropout: float


@dataclass(frozen=True, slots=True)
class FinalRole:
    manifest: PinnedFile
    expected_rows: int
    expected_pairs: int
    pairs_by_language: dict[str, int]
    sources_by_language: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class Runtime:
    python_version: str
    torch_version: str
    cuda_runtime: str
    transformers_version: str


@dataclass(frozen=True, slots=True)
class Inference:
    sample_rate: int
    window_samples: int
    batch_size: int
    num_workers: int


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
    authorization: PinnedFile
    license_ledger: PinnedFile
    protocol: dict[str, str]
    evidence: dict[str, PinnedFile]
    checkpoint: Checkpoint
    encoder: Encoder
    head: Head
    final: FinalRole
    implementation: tuple[PinnedFile, ...]
    runtime: Runtime
    inference: Inference
    outputs: Outputs


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4FinalEvaluationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    return _object(value, label)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        raise V4FinalEvaluationError(
            f"{label} fields differ; missing={missing!r}, unknown={unknown!r}."
        )


def _string(raw: Mapping[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise V4FinalEvaluationError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4FinalEvaluationError(f"{label} must be a positive integer.")
    return value


def _relative_path(value: str, base: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise V4FinalEvaluationError(f"{label} must be relative to the plan.")
    candidate = (base / path).resolve()
    if not candidate.is_relative_to(base.parents[2]):
        raise V4FinalEvaluationError(f"{label} escapes the project root.")
    return candidate


def _pinned(value: object, label: str, base: Path) -> PinnedFile:
    raw = _object(value, label)
    _exact_keys(raw, {"path", "sha256"}, label)
    digest = _string(raw, "sha256", label)
    if _SHA256.fullmatch(digest) is None:
        raise V4FinalEvaluationError(f"{label}.sha256 must be a lowercase SHA-256 digest.")
    return PinnedFile(_relative_path(_string(raw, "path", label), base, label), digest)


def _verify(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file() or sha256_file(pinned.path) != pinned.sha256:
        raise V4FinalEvaluationError(f"Pinned {label} is missing or changed: {pinned.path}")


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalEvaluationError(f"Cannot read {label}: {path}") from error


def _load_checkpoint(value: object, base: Path) -> Checkpoint:
    raw = _object(value, "checkpoint")
    _exact_keys(raw, {"path", "sha256", "selected_model_state_sha256"}, "checkpoint")
    digest = _string(raw, "sha256", "checkpoint")
    state_digest = _string(raw, "selected_model_state_sha256", "checkpoint")
    if _SHA256.fullmatch(digest) is None or _SHA256.fullmatch(state_digest) is None:
        raise V4FinalEvaluationError("checkpoint digests are invalid.")
    return Checkpoint(
        path=_relative_path(_string(raw, "path", "checkpoint"), base, "checkpoint"),
        sha256=digest,
        selected_state_sha256=state_digest,
    )


def _load_encoder(value: object, base: Path) -> Encoder:
    raw = _object(value, "encoder")
    _exact_keys(raw, {"checkpoint_dir", "revision", "config", "weights"}, "encoder")
    directory = _relative_path(_string(raw, "checkpoint_dir", "encoder"), base, "encoder")
    if not directory.is_dir():
        raise V4FinalEvaluationError(f"Pinned XLS-R encoder directory is missing: {directory}")
    return Encoder(
        directory=directory,
        revision=_string(raw, "revision", "encoder"),
        config=_pinned(raw["config"], "encoder.config", base),
        weights=_pinned(raw["weights"], "encoder.weights", base),
    )


def _load_head(value: object) -> Head:
    raw = _object(value, "head")
    _exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    dropout = raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise V4FinalEvaluationError("head.dropout must be in [0, 1).")
    return Head(
        attention_size=_positive_int(raw.get("attention_size"), "head.attention_size"),
        classifier_size=_positive_int(raw.get("classifier_size"), "head.classifier_size"),
        dropout=float(dropout),
    )


def _load_final(value: object, base: Path) -> FinalRole:
    raw = _object(value, "final")
    _exact_keys(
        raw,
        {
            "manifest",
            "selected_split",
            "expected_rows",
            "expected_pairs",
            "pairs_by_language",
            "sources_by_language",
        },
        "final",
    )
    if _string(raw, "selected_split", "final") != "test":
        raise V4FinalEvaluationError("final.selected_split must be 'test'.")
    pairs_by_language_raw = _object(raw["pairs_by_language"], "final.pairs_by_language")
    _exact_keys(pairs_by_language_raw, {"ru", "kk"}, "final.pairs_by_language")
    pairs_by_language = {
        language: _positive_int(
            pairs_by_language_raw[language], f"final.pairs_by_language.{language}"
        )
        for language in ("ru", "kk")
    }
    sources_raw = _object(raw["sources_by_language"], "final.sources_by_language")
    _exact_keys(sources_raw, {"ru", "kk"}, "final.sources_by_language")
    sources: dict[str, dict[str, str]] = {}
    for language in ("ru", "kk"):
        labels = _object(sources_raw[language], f"final.sources_by_language.{language}")
        _exact_keys(labels, {"bonafide", "spoof"}, f"final.sources_by_language.{language}")
        sources[language] = {
            label: _string(labels, label, f"final.sources_by_language.{language}")
            for label in ("bonafide", "spoof")
        }
    rows = _positive_int(raw.get("expected_rows"), "final.expected_rows")
    pairs = _positive_int(raw.get("expected_pairs"), "final.expected_pairs")
    if rows != 2 * pairs or pairs != sum(pairs_by_language.values()):
        raise V4FinalEvaluationError("final counts are not a complete bilingual pair lock.")
    return FinalRole(
        manifest=_pinned(raw["manifest"], "final.manifest", base),
        expected_rows=rows,
        expected_pairs=pairs,
        pairs_by_language=pairs_by_language,
        sources_by_language=sources,
    )


def _load_runtime(value: object) -> Runtime:
    raw = _object(value, "runtime")
    _exact_keys(
        raw, {"python_version", "torch_version", "cuda_runtime", "transformers_version"}, "runtime"
    )
    return Runtime(
        python_version=_string(raw, "python_version", "runtime"),
        torch_version=_string(raw, "torch_version", "runtime"),
        cuda_runtime=_string(raw, "cuda_runtime", "runtime"),
        transformers_version=_string(raw, "transformers_version", "runtime"),
    )


def _load_inference(value: object) -> Inference:
    raw = _object(value, "inference")
    _exact_keys(
        raw,
        {"sample_rate", "window_samples", "batch_size", "num_workers", "device", "precision"},
        "inference",
    )
    workers = raw.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise V4FinalEvaluationError("inference.num_workers must be non-negative.")
    if (
        _string(raw, "device", "inference") != "cuda"
        or _string(raw, "precision", "inference") != "bf16"
    ):
        raise V4FinalEvaluationError("final inference requires CUDA BF16.")
    return Inference(
        sample_rate=_positive_int(raw.get("sample_rate"), "inference.sample_rate"),
        window_samples=_positive_int(raw.get("window_samples"), "inference.window_samples"),
        batch_size=_positive_int(raw.get("batch_size"), "inference.batch_size"),
        num_workers=workers,
    )


def _load_outputs(value: object, base: Path) -> Outputs:
    raw = _object(value, "outputs")
    _exact_keys(raw, {"preflight", "execution_lock", "report"}, "outputs")
    outputs = Outputs(
        preflight=_relative_path(_string(raw, "preflight", "outputs"), base, "outputs.preflight"),
        execution_lock=_relative_path(
            _string(raw, "execution_lock", "outputs"), base, "outputs.execution_lock"
        ),
        report=_relative_path(_string(raw, "report", "outputs"), base, "outputs.report"),
    )
    if len({outputs.preflight, outputs.execution_lock, outputs.report}) != 3:
        raise V4FinalEvaluationError("final evaluation outputs must be distinct.")
    if any(
        not output.parent.is_dir()
        for output in (outputs.preflight, outputs.execution_lock, outputs.report)
    ):
        raise V4FinalEvaluationError("final evaluation output directories must already exist.")
    return outputs


def load_v4_final_evaluation_plan(path: Path) -> Plan:
    """Parse the contract and verify every versioned byte before CUDA is touched."""

    if not path.is_file():
        raise V4FinalEvaluationError(f"Final evaluation plan is missing: {path}")
    try:
        plan_bytes = path.read_bytes()
        raw = _object(json.loads(plan_bytes), "final evaluation plan")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalEvaluationError(f"Cannot read final evaluation plan: {path}") from error
    _exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "run_id",
            "purpose",
            "authorization",
            "protocol",
            "license_ledger",
            "evidence",
            "checkpoint",
            "encoder",
            "head",
            "final",
            "implementation",
            "runtime",
            "inference",
            "outputs",
        },
        "final evaluation plan",
    )
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or _string(raw, "protocol_id", "final evaluation plan") != PROTOCOL_ID
    ):
        raise V4FinalEvaluationError("final evaluation schema/protocol is invalid.")
    if _string(raw, "purpose", "final evaluation plan") != "research":
        raise V4FinalEvaluationError("final evaluation is personal research only.")
    run_id = _string(raw, "run_id", "final evaluation plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise V4FinalEvaluationError("final evaluation run_id is invalid.")
    protocol = _object(raw["protocol"], "protocol")
    if protocol != _PROTOCOL:
        raise V4FinalEvaluationError(
            "final evaluation protocol limits differ from the fixed contract."
        )
    evidence_raw = _object(raw["evidence"], "evidence")
    expected_evidence = {
        "pair_lock_receipt",
        "pair_lock_authorization",
        "reconciliation_receipt",
        "training_plan",
        "training_receipt",
        "train_manifest",
        "dev_manifest",
        "calibration_plan",
        "calibration_receipt",
        "calibration_manifest",
    }
    _exact_keys(evidence_raw, expected_evidence, "evidence")
    implementation_raw = raw["implementation"]
    if not isinstance(implementation_raw, list) or not implementation_raw:
        raise V4FinalEvaluationError("implementation must be a non-empty array.")
    base = path.resolve().parent
    plan = Plan(
        run_id=run_id,
        path=path.resolve(),
        sha256=hashlib.sha256(plan_bytes).hexdigest(),
        authorization=_pinned(raw["authorization"], "authorization", base),
        license_ledger=_pinned(raw["license_ledger"], "license_ledger", base),
        protocol=cast(dict[str, str], protocol),
        evidence={
            name: _pinned(evidence_raw[name], f"evidence.{name}", base)
            for name in sorted(expected_evidence)
        },
        checkpoint=_load_checkpoint(raw["checkpoint"], base),
        encoder=_load_encoder(raw["encoder"], base),
        head=_load_head(raw["head"]),
        final=_load_final(raw["final"], base),
        implementation=tuple(
            _pinned(item, "implementation item", base) for item in implementation_raw
        ),
        runtime=_load_runtime(raw["runtime"]),
        inference=_load_inference(raw["inference"]),
        outputs=_load_outputs(raw["outputs"], base),
    )
    if len({item.path for item in plan.implementation}) != len(plan.implementation):
        raise V4FinalEvaluationError("implementation paths must be unique.")
    for pinned, label in (
        (plan.authorization, "authorization"),
        (plan.license_ledger, "license ledger"),
        *[(value, f"evidence.{name}") for name, value in plan.evidence.items()],
        (plan.encoder.config, "XLS-R config"),
        (plan.encoder.weights, "XLS-R weights"),
        (plan.final.manifest, "final pair manifest"),
        *[(item, "implementation") for item in plan.implementation],
    ):
        _verify(pinned, label)
    _validate_authorization(plan)
    _validate_evidence(plan)
    return plan


def _validate_authorization(plan: Plan) -> None:
    raw = _read_json(plan.authorization.path, "final evaluation authorization")
    _exact_keys(
        raw,
        {"schema_version", "protocol_id", "created_at", "authority", "grants", "prohibitions"},
        "final evaluation authorization",
    )
    grants = _object(raw.get("grants"), "authorization grants")
    prohibitions = _object(raw.get("prohibitions"), "authorization prohibitions")
    if (
        raw.get("schema_version") != 1
        or raw.get("protocol_id")
        != "xlsr-sls-model-v4-final-reconciliation-evaluation-authorization-v1"
        or _string(raw, "authority", "final evaluation authorization")
        != "project_owner_active_session"
        or grants
        != {
            "checkpoint_loading": True,
            "one_time_final_inference": True,
            "ru_temperature_reuse": True,
        }
        or prohibitions
        != {
            "calibration_refit": True,
            "checkpoint_mutation": True,
            "network_downloads": True,
            "output_overwrite": True,
            "pair_mutation_or_backfill": True,
            "retraining": True,
            "threshold_selection": True,
        }
    ):
        raise V4FinalEvaluationError("final evaluation authorization is not exact and fail-closed.")


def _validate_evidence(plan: Plan) -> None:
    pair_lock = _read_json(plan.evidence["pair_lock_receipt"].path, "pair lock receipt")
    pair_claims = _object(pair_lock.get("claims"), "pair lock claims")
    pair_counts = _object(pair_lock.get("counts"), "pair lock counts")
    output = _object(pair_lock.get("output"), "pair lock output")
    if (
        pair_lock.get("status") != "pair_lock_complete_final_inference_still_forbidden"
        or pair_claims.get("pair_lock_performed") is not True
        or pair_claims.get("independent_acoustic_language_review_performed") is not True
        or pair_claims.get("final_inference_performed") is not False
        or pair_counts.get("locked_assets") != plan.final.expected_rows
        or pair_counts.get("locked_pairs") != plan.final.expected_pairs
        or pair_counts.get("locked_pairs_by_language") != plan.final.pairs_by_language
        or output.get("sha256") != plan.final.manifest.sha256
        or output.get("rows") != plan.final.expected_rows
    ):
        raise V4FinalEvaluationError("pair-lock receipt does not bind the final evaluation input.")
    reconciliation = _read_json(
        plan.evidence["reconciliation_receipt"].path, "reconciliation receipt"
    )
    reconciliation_claims = _object(reconciliation.get("claims"), "reconciliation claims")
    if (
        reconciliation.get("status") != "materialized_review_required_pair_lock_pending"
        or reconciliation_claims.get("full_history_audio_isolation_performed") is not True
        or reconciliation_claims.get("detector_inference_performed") is not False
        or reconciliation_claims.get("final_inference_performed") is not False
    ):
        raise V4FinalEvaluationError(
            "reconciliation receipt does not establish pre-logit isolation."
        )
    training = _read_json(plan.evidence["training_receipt"].path, "training receipt")
    if (
        training.get("status") != "ok"
        or training.get("selected_model_state_sha256") != plan.checkpoint.selected_state_sha256
        or training.get("calibrated") is not False
        or training.get("final_inference_performed") is not False
    ):
        raise V4FinalEvaluationError(
            "training receipt does not identify the selected uncalibrated v4 checkpoint."
        )
    calibration = _read_json(plan.evidence["calibration_receipt"].path, "calibration receipt")
    calibration_data = _object(calibration.get("calibration"), "calibration data")
    temperature = calibration_data.get("temperature")
    checkpoint = _object(calibration.get("frozen_checkpoint"), "calibration frozen checkpoint")
    if (
        calibration.get("status") != "ok"
        or calibration.get("temperature_fitted") is not True
        or calibration.get("threshold_selection_performed") is not False
        or calibration.get("final_inference_performed") is not False
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
        or checkpoint.get("sha256") != plan.checkpoint.sha256
        or checkpoint.get("selected_model_state_sha256") != plan.checkpoint.selected_state_sha256
    ):
        raise V4FinalEvaluationError(
            "RU calibration receipt is not reusable for this final checkpoint."
        )


def validate_v4_final_evaluation_inputs(
    plan: Plan, ledger: Mapping[str, LicenseLedgerEntry]
) -> tuple[tuple[ManifestRow, ...], dict[str, dict[str, int]]]:
    """Validate pair completeness and zero detectable train/dev/calibration reuse."""

    final_rows = tuple(load_manifest(plan.final.manifest.path))
    issues: list[str] = []
    try:
        validate_manifest(final_rows)
        validate_manifest_licenses(final_rows, ledger)
    except (ManifestError, LicenseLedgerError) as error:
        issues.extend(error.issues)
    if len(final_rows) != plan.final.expected_rows:
        issues.append("final row count differs from the immutable plan.")
    if Counter(row.label for row in final_rows) != Counter(
        {"bonafide": plan.final.expected_pairs, "spoof": plan.final.expected_pairs}
    ):
        issues.append("final labels are not balanced by the immutable pair count.")
    if any(row.split != "test" for row in final_rows):
        issues.append("final manifest contains a non-test row.")
    observed_pairs_by_language: dict[str, int] = {}
    for language in ("ru", "kk"):
        rows = [row for row in final_rows if row.language == language]
        pairs: dict[str, list[ManifestRow]] = defaultdict(list)
        for row in rows:
            pairs[row.text_id].append(row)
            expected_source = plan.final.sources_by_language[language].get(row.label)
            if row.source_name != expected_source:
                issues.append(
                    f"final {language}/{row.label} source differs from the immutable plan."
                )
        observed_pairs_by_language[language] = len(pairs)
        if len(pairs) != plan.final.pairs_by_language[language] or any(
            len(pair) != 2
            or {item.label for item in pair} != {"bonafide", "spoof"}
            or len({item.text_hash for item in pair}) != 1
            or len({item.language for item in pair}) != 1
            for pair in pairs.values()
        ):
            issues.append(f"final {language} rows are not complete exact bonafide/spoof pairs.")
    if any(row.language not in {"ru", "kk"} or row.code_switch != "false" for row in final_rows):
        issues.append("final rows must be non-code-switch RU/KK assets only.")
    expected_sources = {
        source for labels in plan.final.sources_by_language.values() for source in labels.values()
    }
    if {row.source_name for row in final_rows} != expected_sources:
        issues.append("final manifest source set differs from the immutable plan.")
    for source in expected_sources:
        entry = ledger.get(source)
        if entry is None or entry.train_dev_test_use not in {"research_only", "product_allowed"}:
            issues.append(f"final source {source!r} is not permitted for this research test role.")
    other_roles: dict[str, tuple[ManifestRow, ...]] = {}
    for name in ("train_manifest", "dev_manifest", "calibration_manifest"):
        try:
            other_roles[name] = tuple(load_manifest(plan.evidence[name].path))
            validate_manifest(other_roles[name])
        except ManifestError as error:
            issues.extend(f"{name}: {issue}" for issue in error.issues)
            other_roles[name] = ()
    overlap_counts: dict[str, dict[str, int]] = {}
    for role, role_rows in other_roles.items():
        fields = ("sample_id", "sha256", "text_hash", "parent_group_id", "source_name")
        overlap_counts[role] = {
            field: len(
                {getattr(row, field) for row in final_rows}.intersection(
                    getattr(row, field) for row in role_rows
                )
            )
            for field in fields
        }
        if any(overlap_counts[role].values()):
            issues.append(f"final/{role} detectable leakage is non-zero: {overlap_counts[role]!r}.")
    if issues:
        raise V4FinalEvaluationError(issues)
    return final_rows, overlap_counts


def validate_v4_final_evaluation_checkpoint_file(plan: Plan) -> None:
    """Validate checkpoint bytes without deserializing the model."""

    if (
        not plan.checkpoint.path.is_file()
        or sha256_file(plan.checkpoint.path) != plan.checkpoint.sha256
    ):
        raise V4FinalEvaluationError("selected v4 checkpoint is missing or changed.")


def _cuda_device(plan: Plan) -> torch.device:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("v4 final evaluation requires an available CUDA BF16 device.")
    actual = Runtime(
        platform.python_version(),
        torch.__version__,
        str(torch.version.cuda),
        transformers.__version__,
    )
    if actual != plan.runtime:
        raise RuntimeError(
            f"v4 final runtime lock mismatch: expected {plan.runtime!r}, got {actual!r}."
        )
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


def _record(pinned: PinnedFile) -> dict[str, str]:
    return {"path": str(pinned.path), "sha256": pinned.sha256}


def plan_record(plan: Plan) -> dict[str, object]:
    """Return a receipt-safe serialization of every execution-relevant binding."""

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.path),
        "plan_sha256": plan.sha256,
        "authorization": _record(plan.authorization),
        "license_ledger": _record(plan.license_ledger),
        "protocol": plan.protocol,
        "evidence": {name: _record(value) for name, value in plan.evidence.items()},
        "checkpoint": {
            "path": str(plan.checkpoint.path),
            "sha256": plan.checkpoint.sha256,
            "selected_model_state_sha256": plan.checkpoint.selected_state_sha256,
        },
        "encoder": {
            "checkpoint_dir": str(plan.encoder.directory),
            "revision": plan.encoder.revision,
            "config": _record(plan.encoder.config),
            "weights": _record(plan.encoder.weights),
        },
        "head": asdict(plan.head),
        "final": {
            "manifest": _record(plan.final.manifest),
            "selected_split": "test",
            "expected_rows": plan.final.expected_rows,
            "expected_pairs": plan.final.expected_pairs,
            "pairs_by_language": plan.final.pairs_by_language,
            "sources_by_language": plan.final.sources_by_language,
        },
        "implementation": [_record(item) for item in plan.implementation],
        "runtime": asdict(plan.runtime),
        "inference": {
            **asdict(plan.inference),
            "device": "cuda",
            "precision": "bf16",
            "fixed_decision_boundary": "raw_logit_zero",
        },
        "outputs": {
            "preflight": str(plan.outputs.preflight),
            "execution_lock": str(plan.outputs.execution_lock),
            "report": str(plan.outputs.report),
        },
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
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise V4FinalEvaluationError(f"Refusing to overwrite output: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preflight_v4_final_evaluation(
    plan: Plan,
    rows: Sequence[ManifestRow],
    overlap_counts: Mapping[str, Mapping[str, int]],
    device: torch.device,
) -> dict[str, object]:
    """Create the no-logit readiness receipt after final-asset hashes are verified."""

    return {
        "status": "validated",
        "mode": "no_logit_preflight",
        "run_plan": plan_record(plan),
        "assets_validated": len(rows),
        "final_pairs_validated": plan.final.expected_pairs,
        "pairwise_overlap_counts": {role: dict(counts) for role, counts in overlap_counts.items()},
        "environment": _environment(device),
        "checkpoint_file_verified": True,
        "checkpoint_loaded": False,
        "final_inference_performed": False,
        "threshold_selection_performed": False,
        "calibration_refit_performed": False,
        "detector_feedback": False,
    }


def write_v4_final_evaluation_preflight(plan: Plan, preflight: Mapping[str, object]) -> str:
    """Publish a no-logit preflight exactly once and return its digest."""

    _write_exclusive_json(plan.outputs.preflight, preflight)
    return sha256_file(plan.outputs.preflight)


def _require_preflight(plan: Plan) -> None:
    receipt = _read_json(plan.outputs.preflight, "final-evaluation preflight")
    run_plan = _object(receipt.get("run_plan"), "preflight run plan")
    if (
        receipt.get("status") != "validated"
        or receipt.get("mode") != "no_logit_preflight"
        or receipt.get("assets_validated") != plan.final.expected_rows
        or receipt.get("final_pairs_validated") != plan.final.expected_pairs
        or receipt.get("checkpoint_file_verified") is not True
        or receipt.get("checkpoint_loaded") is not False
        or receipt.get("final_inference_performed") is not False
        or receipt.get("threshold_selection_performed") is not False
        or receipt.get("calibration_refit_performed") is not False
        or receipt.get("detector_feedback") is not False
        or run_plan.get("protocol_id") != PROTOCOL_ID
        or run_plan.get("plan_sha256") != plan.sha256
    ):
        raise V4FinalEvaluationError("no-logit preflight does not bind this final evaluation plan.")


def _load_state(plan: Plan) -> dict[str, Tensor]:
    validate_v4_final_evaluation_checkpoint_file(plan)
    with torch.serialization.safe_globals([TorchVersion]):
        value: object = torch.load(plan.checkpoint.path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise V4FinalEvaluationError("v4 checkpoint root must be a dictionary.")
    checkpoint = cast(dict[str, object], value)
    if (
        checkpoint.get("model_name") != "xlsr_sls_model_v4"
        or checkpoint.get("training_purpose") != "research"
        or checkpoint.get("run_id") != "xlsr-sls-model-v4-train-v1"
        or checkpoint.get("selected_model_state_sha256") != plan.checkpoint.selected_state_sha256
    ):
        raise V4FinalEvaluationError("v4 checkpoint identity does not match this final contract.")
    state_value = checkpoint.get("model_state_dict")
    if not isinstance(state_value, dict) or not state_value:
        raise V4FinalEvaluationError("v4 checkpoint has no model_state_dict.")
    state = cast(dict[str, Tensor], state_value)
    if any(
        not isinstance(key, str) or not isinstance(tensor, Tensor) for key, tensor in state.items()
    ):
        raise V4FinalEvaluationError("v4 model_state_dict has invalid entries.")
    if state_dict_sha256(state) != plan.checkpoint.selected_state_sha256:
        raise V4FinalEvaluationError(
            "v4 checkpoint state digest does not match this final contract."
        )
    return state


def _build_model(
    plan: Plan, state: Mapping[str, Tensor], device: torch.device
) -> XlsrSlsClassifier:
    model = XlsrSlsClassifier.from_pretrained(
        str(plan.encoder.directory),
        attention_size=plan.head.attention_size,
        classifier_size=plan.head.classifier_size,
        dropout=plan.head.dropout,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise V4FinalEvaluationError(
            "v4 checkpoint could not be loaded strictly into the pinned model."
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
                    raise RuntimeError(f"Final inference duplicated sample_id={sample_id!r}.")
                logits_by_id[sample_id] = float(logit)
                labels_by_id[sample_id] = float(label)
    if set(logits_by_id) != {row.sample_id for row in rows}:
        raise RuntimeError("Final inference did not produce exactly one logit per frozen row.")
    return torch.tensor([logits_by_id[row.sample_id] for row in rows]), torch.tensor(
        [labels_by_id[row.sample_id] for row in rows]
    )


def _metric(correct: int, examples: int) -> dict[str, object]:
    return {
        "correct": correct,
        "examples": examples,
        "value": correct / examples,
        "confidence_interval": asdict(wilson_interval(correct, examples)),
    }


def _classification_metrics(scores: Tensor, labels: Tensor) -> dict[str, object]:
    predictions = scores >= 0
    expected = labels >= 0.5
    bona = ~expected
    spoof = expected
    bona_correct = int((predictions[bona] == expected[bona]).sum())
    spoof_correct = int((predictions[spoof] == expected[spoof]).sum())
    bona_count = int(bona.sum())
    spoof_count = int(spoof.sum())
    return {
        "decision_boundary": "raw_logit_zero",
        "accuracy": _metric(int((predictions == expected).sum()), int(labels.numel())),
        "bonafide_recall": _metric(bona_correct, bona_count),
        "spoof_recall": _metric(spoof_correct, spoof_count),
        "balanced_accuracy": (bona_correct / bona_count + spoof_correct / spoof_count) / 2,
    }


def _eer(scores: Tensor, labels: Tensor) -> float:
    """Return deterministic empirical EER; it is descriptive, never an operating threshold."""

    values = sorted({float(value) for value in scores.tolist()}, reverse=True)
    thresholds = [float("inf"), *values, float("-inf")]
    positive = labels >= 0.5
    negative = ~positive
    candidates: list[tuple[float, float]] = []
    for threshold in thresholds:
        prediction = scores >= threshold
        false_positive = float((prediction[negative]).sum()) / int(negative.sum())
        false_negative = float((~prediction[positive]).sum()) / int(positive.sum())
        candidates.append(
            (abs(false_positive - false_negative), (false_positive + false_negative) / 2)
        )
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _strata(
    rows: Sequence[ManifestRow], scores: Tensor, labels: Tensor, field: str
) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = getattr(row, field) or "not_applicable"
        groups[value].append(index)
    result: dict[str, object] = {}
    for value, indices in sorted(groups.items()):
        group_labels = labels[indices]
        group_scores = scores[indices]
        expected = group_labels >= 0.5
        predicted = group_scores >= 0
        labels_count = Counter(rows[index].label for index in indices)
        record: dict[str, object] = {
            "records": len(indices),
            "label_counts": dict(sorted(labels_count.items())),
            "decision_accuracy": _metric(int((predicted == expected).sum()), len(indices)),
        }
        if len(labels_count) == 2:
            record["classification"] = _classification_metrics(group_scores, group_labels)
            record["eer"] = _eer(group_scores, group_labels)
        result[value] = record
    return result


def _language_report(
    rows: Sequence[ManifestRow], logits: Tensor, ru_temperature: float | None
) -> dict[str, object]:
    labels = torch.tensor([1.0 if row.label == "spoof" else 0.0 for row in rows])
    if not torch.equal(
        labels, torch.tensor([1.0 if row.label == "spoof" else 0.0 for row in rows])
    ):
        raise RuntimeError("Final labels differ from the frozen manifest.")
    prediction = logits >= 0
    pair_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair_indices[row.text_id].append(index)
    both_correct = sum(
        bool((prediction[indices] == (labels[indices] >= 0.5)).all())
        for indices in pair_indices.values()
    )
    sample_results: list[dict[str, object]] = []
    calibrated_probabilities = torch.sigmoid(logits / ru_temperature) if ru_temperature else None
    for index, row in enumerate(rows):
        result: dict[str, object] = {
            "sample_id": row.sample_id,
            "label": row.label,
            "source_name": row.source_name,
            "generator_family": row.generator_family or "not_applicable",
            "voice_id": row.voice_id or "not_applicable",
            "text_id": row.text_id,
            "audio_sha256": row.sha256,
            "raw_logit": float(logits[index]),
            "prediction_at_fixed_raw_logit_zero": "spoof"
            if bool(prediction[index])
            else "bonafide",
        }
        if calibrated_probabilities is not None:
            result["ru_temperature_scaled_spoof_probability"] = float(
                calibrated_probabilities[index]
            )
        sample_results.append(result)
    report: dict[str, object] = {
        "records": len(rows),
        "pairs": len(pair_indices),
        "classification": _classification_metrics(logits, labels),
        "eer": _eer(logits, labels),
        "eer_method": "empirical_descriptive_not_an_operating_threshold",
        "pairs_both_correct": _metric(both_correct, len(pair_indices)),
        "source_metrics": _strata(rows, logits, labels, "source_name"),
        "generator_family_metrics": _strata(rows, logits, labels, "generator_family"),
        "voice_control_metrics": _strata(rows, logits, labels, "voice_id"),
        "sample_results": sample_results,
    }
    if calibrated_probabilities is not None:
        report["ru_calibrated_probability_diagnostics"] = {
            "temperature": ru_temperature,
            "brier_score": brier_score(calibrated_probabilities, labels),
            "expected_calibration_error_15_bins": expected_calibration_error(
                calibrated_probabilities, labels
            ),
            "scope": (
                "RU-only; fixed temperature reused without refit and not a product "
                "probability claim"
            ),
        }
    return report


def execute_v4_final_evaluation(
    plan: Plan,
    rows: Sequence[ManifestRow],
    device: torch.device,
    audio_root: Path,
    preflight: Mapping[str, object],
) -> dict[str, object]:
    """Run the sole authorized final scoring pass and publish its immutable receipt."""

    _require_preflight(plan)
    existing = [
        str(path) for path in (plan.outputs.execution_lock, plan.outputs.report) if path.exists()
    ]
    if existing:
        raise V4FinalEvaluationError("Refusing repeated final inference: " + ", ".join(existing))
    execution_lock = {
        **preflight,
        "status": "final_inference_started",
        "mode": "one_time_gpu_final_evaluation",
        "preflight": {
            "path": str(plan.outputs.preflight),
            "sha256": sha256_file(plan.outputs.preflight),
        },
        "one_time_execution": True,
        "checkpoint_loaded": True,
        "final_inference_performed": True,
        "calibration_refit_performed": False,
        "threshold_selection_performed": False,
        "report_path": str(plan.outputs.report),
    }
    _write_exclusive_json(plan.outputs.execution_lock, execution_lock)
    torch.cuda.reset_peak_memory_stats(device)
    state = _load_state(plan)
    model = _build_model(plan, state, device)
    by_language = {
        language: tuple(row for row in rows if row.language == language)
        for language in ("ru", "kk")
    }
    calibration = _read_json(plan.evidence["calibration_receipt"].path, "calibration receipt")
    temperature_value = _object(calibration["calibration"], "calibration data").get("temperature")
    if not isinstance(temperature_value, (int, float)) or isinstance(temperature_value, bool):
        raise V4FinalEvaluationError("calibration receipt has no numeric RU temperature.")
    ru_temperature = float(temperature_value)
    reports: dict[str, object] = {}
    for language in ("ru", "kk"):
        logits, labels = _infer_logits(plan, by_language[language], model, device, audio_root)
        expected = torch.tensor(
            [1.0 if row.label == "spoof" else 0.0 for row in by_language[language]]
        )
        if not torch.equal(labels, expected):
            raise RuntimeError(f"Final dataset labels differ from manifest for {language}.")
        reports[language] = _language_report(
            by_language[language], logits, ru_temperature if language == "ru" else None
        )
    torch.cuda.synchronize(device)
    report = {
        **preflight,
        "status": "ok",
        "mode": "one_time_gpu_final_evaluation",
        "execution_lock": _record(
            PinnedFile(plan.outputs.execution_lock, sha256_file(plan.outputs.execution_lock))
        ),
        "frozen_checkpoint": {
            "path": str(plan.checkpoint.path),
            "sha256": plan.checkpoint.sha256,
            "selected_model_state_sha256": plan.checkpoint.selected_state_sha256,
        },
        "final_by_language": reports,
        "pooled_ru_kk_metric": "prohibited",
        "checkpoint_loaded": True,
        "final_inference_performed": True,
        "threshold_selection_performed": False,
        "calibration_refit_performed": False,
        "detector_feedback": False,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "limitations": [
            (
                "Personal research only; this is not product quality, a fraud-risk score, or a "
                "deployment decision."
            ),
            (
                "The final set is isolated by available source/text/group/audio evidence, but "
                "speaker independence is not verified."
            ),
            (
                "RU temperature was fitted earlier on a disjoint 73-pair calibration role; it was "
                "reused without refit."
            ),
            "KK output remains an uncalibrated score; no KK probability claim is made.",
            (
                "No pooled RU+KK headline, threshold selection, retraining, backfill, or repeat "
                "final inference is permitted."
            ),
        ],
    }
    _write_exclusive_json(plan.outputs.report, report)
    return report
