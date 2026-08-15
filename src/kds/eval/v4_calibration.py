"""Fail-closed execution primitives for the XLS-R+SLS model-v4 RU calibration run.

The v4 training checkpoint is selected already.  This module permits exactly one
additional operation: score the separately frozen Russian calibration pairs and
fit the existing one-parameter temperature scaler.  It deliberately contains no
final-set role and no threshold-search interface.
"""

from __future__ import annotations

import hashlib
import json
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
from kds.data.licenses import LicenseLedgerEntry, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.eval.calibration import CalibrationReport, TemperatureScaler
from kds.models import XlsrSlsClassifier
from kds.training import make_audio_loader
from kds.training.frozen_b0 import state_dict_sha256

V4_CALIBRATION_SCHEMA_VERSION = 1
V4_CALIBRATION_PROTOCOL_ID = "xlsr-sls-model-v4-ru-calibration-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_PROTOCOL = {
    "kind": "one_time_ru_temperature_calibration_after_pair_lock",
    "purpose": "personal_research_only",
    "checkpoint_selection": "already_fixed_to_v4_macro_ru_kk_dev_loss",
    "calibration": "temperature_scaling_only_on_frozen_ru_pairs",
    "calibration_language_scope": "ru_only",
    "threshold_selection": "prohibited",
    "final_inference": "prohibited",
    "detector_feedback": "prohibited",
    "checkpoint_mutation": "prohibited",
    "pair_mutation_or_backfill": "prohibited",
}


class V4CalibrationError(ValueError):
    """Raised when a calibration input or immutable execution state is invalid."""

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
class LabelCounts:
    bonafide: int
    spoof: int

    def as_dict(self) -> dict[str, int]:
        return {"bonafide": self.bonafide, "spoof": self.spoof}


@dataclass(frozen=True, slots=True)
class CalibrationRole:
    manifest: PinnedFile
    expected_rows: int
    expected_pairs: int
    expected_labels: LabelCounts
    expected_sources: tuple[str, ...]


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
    temperature_max_iter: int


@dataclass(frozen=True, slots=True)
class Outputs:
    preflight: Path
    execution_lock: Path
    report: Path


@dataclass(frozen=True, slots=True)
class Evidence:
    materialization_plan: PinnedFile
    materialization_receipt: PinnedFile
    training_plan: PinnedFile
    training_receipt: PinnedFile
    train_manifest: PinnedFile
    dev_manifest: PinnedFile


@dataclass(frozen=True, slots=True)
class Plan:
    run_id: str
    path: Path
    sha256: str
    protocol: dict[str, str]
    license_ledger: PinnedFile
    evidence: Evidence
    checkpoint: Checkpoint
    encoder: Encoder
    head: Head
    calibration: CalibrationRole
    implementation: tuple[PinnedFile, ...]
    runtime: Runtime
    inference: Inference
    outputs: Outputs


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4CalibrationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    if missing or unknown:
        raise V4CalibrationError(
            f"{label} fields differ; missing={missing!r}, unknown={unknown!r}."
        )


def _string(raw: Mapping[str, object], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise V4CalibrationError(f"{label}.{key} must be a non-empty string.")
    return value.strip()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4CalibrationError(f"{label} must be a positive integer.")
    return value


def _relative_path(value: str, base: Path, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise V4CalibrationError(f"{label} must be relative to the plan.")
    candidate = (base / path).resolve()
    if not candidate.is_relative_to(base.parents[2]):
        raise V4CalibrationError(f"{label} escapes the project root.")
    return candidate


def _pinned(value: object, label: str, base: Path) -> PinnedFile:
    raw = _object(value, label)
    _exact_keys(raw, {"path", "sha256"}, label)
    digest = _string(raw, "sha256", label)
    if _SHA256.fullmatch(digest) is None:
        raise V4CalibrationError(f"{label}.sha256 must be a lowercase SHA-256 digest.")
    return PinnedFile(_relative_path(_string(raw, "path", label), base, label), digest)


def _verify_pinned(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file():
        raise V4CalibrationError(f"{label} is missing: {pinned.path}")
    if sha256_file(pinned.path) != pinned.sha256:
        raise V4CalibrationError(f"{label} SHA-256 differs: {pinned.path}")


def _labels(value: object, label: str) -> LabelCounts:
    raw = _object(value, label)
    _exact_keys(raw, {"bonafide", "spoof"}, label)
    return LabelCounts(
        bonafide=_positive_int(raw.get("bonafide"), f"{label}.bonafide"),
        spoof=_positive_int(raw.get("spoof"), f"{label}.spoof"),
    )


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4CalibrationError(f"Cannot read {label}: {path}") from error


def _load_checkpoint(value: object, base: Path) -> Checkpoint:
    raw = _object(value, "checkpoint")
    _exact_keys(raw, {"path", "sha256", "selected_model_state_sha256"}, "checkpoint")
    digest = _string(raw, "sha256", "checkpoint")
    state_digest = _string(raw, "selected_model_state_sha256", "checkpoint")
    if _SHA256.fullmatch(digest) is None or _SHA256.fullmatch(state_digest) is None:
        raise V4CalibrationError("checkpoint SHA-256 values are invalid.")
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
        raise V4CalibrationError(f"Pinned XLS-R encoder directory is missing: {directory}")
    return Encoder(
        checkpoint_dir=directory,
        revision=_string(raw, "revision", "encoder"),
        config=_pinned(raw["config"], "encoder.config", base),
        weights=_pinned(raw["weights"], "encoder.weights", base),
    )


def _load_head(value: object) -> Head:
    raw = _object(value, "head")
    _exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    dropout = raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise V4CalibrationError("head.dropout must be in [0, 1).")
    return Head(
        attention_size=_positive_int(raw.get("attention_size"), "head.attention_size"),
        classifier_size=_positive_int(raw.get("classifier_size"), "head.classifier_size"),
        dropout=float(dropout),
    )


def _load_role(value: object, base: Path) -> CalibrationRole:
    raw = _object(value, "calibration")
    _exact_keys(
        raw,
        {
            "manifest",
            "selected_split",
            "expected_rows",
            "expected_pairs",
            "expected_label_counts",
            "expected_source_ids",
        },
        "calibration",
    )
    if _string(raw, "selected_split", "calibration") != "test":
        raise V4CalibrationError("v4 frozen calibration pairs must retain split='test'.")
    expected_sources = raw.get("expected_source_ids")
    if not isinstance(expected_sources, list) or len(expected_sources) != 2:
        raise V4CalibrationError(
            "calibration.expected_source_ids must contain exactly two sources."
        )
    sources = tuple(
        sorted(_string({"value": item}, "value", "calibration source") for item in expected_sources)
    )
    if len(set(sources)) != 2:
        raise V4CalibrationError("calibration.expected_source_ids must be distinct.")
    rows = _positive_int(raw.get("expected_rows"), "calibration.expected_rows")
    pairs = _positive_int(raw.get("expected_pairs"), "calibration.expected_pairs")
    labels = _labels(raw["expected_label_counts"], "calibration.expected_label_counts")
    if rows != 2 * pairs or labels != LabelCounts(bonafide=pairs, spoof=pairs):
        raise V4CalibrationError("v4 calibration must contain balanced exact pairs.")
    return CalibrationRole(
        manifest=_pinned(raw["manifest"], "calibration.manifest", base),
        expected_rows=rows,
        expected_pairs=pairs,
        expected_labels=labels,
        expected_sources=sources,
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
        {
            "sample_rate",
            "window_samples",
            "batch_size",
            "num_workers",
            "device",
            "precision",
            "temperature_max_iter",
        },
        "inference",
    )
    workers = raw.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise V4CalibrationError("inference.num_workers must be non-negative.")
    if (
        _string(raw, "device", "inference") != "cuda"
        or _string(raw, "precision", "inference") != "bf16"
    ):
        raise V4CalibrationError("v4 calibration inference requires CUDA BF16.")
    return Inference(
        sample_rate=_positive_int(raw.get("sample_rate"), "inference.sample_rate"),
        window_samples=_positive_int(raw.get("window_samples"), "inference.window_samples"),
        batch_size=_positive_int(raw.get("batch_size"), "inference.batch_size"),
        num_workers=workers,
        temperature_max_iter=_positive_int(
            raw.get("temperature_max_iter"), "inference.temperature_max_iter"
        ),
    )


def _load_outputs(value: object, base: Path) -> Outputs:
    raw = _object(value, "outputs")
    _exact_keys(raw, {"preflight", "execution_lock", "report"}, "outputs")
    outputs = Outputs(
        preflight=_relative_path(_string(raw, "preflight", "outputs"), base, "preflight"),
        execution_lock=_relative_path(
            _string(raw, "execution_lock", "outputs"), base, "execution_lock"
        ),
        report=_relative_path(_string(raw, "report", "outputs"), base, "report"),
    )
    if len({outputs.preflight, outputs.execution_lock, outputs.report}) != 3:
        raise V4CalibrationError("v4 calibration outputs must be distinct.")
    if any(
        not path.parent.is_dir()
        for path in (outputs.preflight, outputs.execution_lock, outputs.report)
    ):
        raise V4CalibrationError("v4 calibration output directories must already exist.")
    return outputs


def load_v4_calibration_plan(path: Path) -> Plan:
    """Load a v4 calibration contract and verify every versioned byte it binds."""

    if not path.is_file():
        raise V4CalibrationError(f"v4 calibration plan is missing: {path}")
    try:
        plan_bytes = path.read_bytes()
        raw = _object(json.loads(plan_bytes), "v4 calibration plan")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4CalibrationError(f"Cannot read v4 calibration plan: {path}") from error
    _exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "run_id",
            "purpose",
            "protocol",
            "license_ledger",
            "evidence",
            "checkpoint",
            "encoder",
            "head",
            "calibration",
            "implementation",
            "runtime",
            "inference",
            "outputs",
        },
        "v4 calibration plan",
    )
    if raw.get("schema_version") != V4_CALIBRATION_SCHEMA_VERSION:
        raise V4CalibrationError("v4 calibration plan schema_version is unsupported.")
    if _string(raw, "protocol_id", "v4 calibration plan") != V4_CALIBRATION_PROTOCOL_ID:
        raise V4CalibrationError("v4 calibration protocol_id is invalid.")
    if _string(raw, "purpose", "v4 calibration plan") != "research":
        raise V4CalibrationError("v4 calibration is research-only.")
    run_id = _string(raw, "run_id", "v4 calibration plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise V4CalibrationError("v4 calibration run_id is invalid.")
    protocol_raw = _object(raw["protocol"], "protocol")
    if protocol_raw != _PROTOCOL:
        raise V4CalibrationError("v4 calibration protocol limits differ from the fixed contract.")
    base = path.resolve().parent
    evidence_raw = _object(raw["evidence"], "evidence")
    _exact_keys(
        evidence_raw,
        {
            "materialization_plan",
            "materialization_receipt",
            "training_plan",
            "training_receipt",
            "train_manifest",
            "dev_manifest",
        },
        "evidence",
    )
    implementation_raw = raw["implementation"]
    if not isinstance(implementation_raw, list) or not implementation_raw:
        raise V4CalibrationError("implementation must be a non-empty array.")
    plan = Plan(
        run_id=run_id,
        path=path.resolve(),
        sha256=hashlib.sha256(plan_bytes).hexdigest(),
        protocol=cast(dict[str, str], protocol_raw),
        license_ledger=_pinned(raw["license_ledger"], "license_ledger", base),
        evidence=Evidence(
            materialization_plan=_pinned(
                evidence_raw["materialization_plan"], "evidence.materialization_plan", base
            ),
            materialization_receipt=_pinned(
                evidence_raw["materialization_receipt"], "evidence.materialization_receipt", base
            ),
            training_plan=_pinned(evidence_raw["training_plan"], "evidence.training_plan", base),
            training_receipt=_pinned(
                evidence_raw["training_receipt"], "evidence.training_receipt", base
            ),
            train_manifest=_pinned(evidence_raw["train_manifest"], "evidence.train_manifest", base),
            dev_manifest=_pinned(evidence_raw["dev_manifest"], "evidence.dev_manifest", base),
        ),
        checkpoint=_load_checkpoint(raw["checkpoint"], base),
        encoder=_load_encoder(raw["encoder"], base),
        head=_load_head(raw["head"]),
        calibration=_load_role(raw["calibration"], base),
        implementation=tuple(
            _pinned(item, "implementation item", base) for item in implementation_raw
        ),
        runtime=_load_runtime(raw["runtime"]),
        inference=_load_inference(raw["inference"]),
        outputs=_load_outputs(raw["outputs"], base),
    )
    if len({item.path for item in plan.implementation}) != len(plan.implementation):
        raise V4CalibrationError("implementation paths must be unique.")
    for item, label in (
        (plan.license_ledger, "v4 calibration ledger"),
        (plan.evidence.materialization_plan, "v4 materialization plan"),
        (plan.evidence.materialization_receipt, "v4 materialization receipt"),
        (plan.evidence.training_plan, "v4 training plan"),
        (plan.evidence.training_receipt, "v4 training receipt"),
        (plan.evidence.train_manifest, "v4 train manifest"),
        (plan.evidence.dev_manifest, "v4 dev manifest"),
        (plan.encoder.config, "XLS-R config"),
        (plan.encoder.weights, "XLS-R weights"),
        (plan.calibration.manifest, "v4 calibration pair lock"),
        *((item, "v4 calibration implementation") for item in plan.implementation),
    ):
        _verify_pinned(item, label)
    _validate_evidence(plan)
    return plan


def _validate_evidence(plan: Plan) -> None:
    materialization_plan = _read_json(
        plan.evidence.materialization_plan.path, "materialization plan"
    )
    materialization_receipt = _read_json(
        plan.evidence.materialization_receipt.path, "materialization receipt"
    )
    training_plan = _read_json(plan.evidence.training_plan.path, "training plan")
    training_receipt = _read_json(plan.evidence.training_receipt.path, "training receipt")
    materialization_outputs = materialization_plan.get("outputs")
    materialization_claims = materialization_receipt.get("claims")
    materialization_counts = materialization_receipt.get("counts")
    if (
        materialization_plan.get("protocol_id")
        != "xlsr-sls-model-v4-calibration-materialization-v1"
        or not isinstance(materialization_outputs, dict)
        or materialization_outputs.get("pair_lock_manifest")
        != "data/manifests/v4/xlsr_sls_model_v4_calibration_pairs_frozen_v1.csv"
        or materialization_receipt.get("state")
        != "ru_calibration_pairs_frozen_checkpoint_scoring_contract_required"
        or not isinstance(materialization_claims, dict)
        or materialization_claims.get("checkpoint_loaded") is not False
        or materialization_claims.get("calibration_performed") is not False
        or materialization_claims.get("temperature_fitted") is not False
        or materialization_claims.get("final_inference_performed") is not False
        or not isinstance(materialization_counts, dict)
        or materialization_counts.get("complete_pairs") != plan.calibration.expected_pairs
        or materialization_counts.get("pair_assets") != plan.calibration.expected_rows
    ):
        raise V4CalibrationError("Materialization evidence does not authorize this pair lock.")
    outputs = materialization_receipt.get("outputs")
    pair_binding = outputs.get("pair_lock_manifest") if isinstance(outputs, dict) else None
    if (
        not isinstance(pair_binding, dict)
        or pair_binding.get("sha256") != plan.calibration.manifest.sha256
        or pair_binding.get("rows") != plan.calibration.expected_rows
    ):
        raise V4CalibrationError(
            "Materialization receipt does not bind the frozen calibration pairs."
        )
    if (
        training_plan.get("protocol_id") != "xlsr-sls-model-v4-training-v1"
        or training_receipt.get("status") != "ok"
        or training_receipt.get("best_tail_unfreeze_epoch") != 2
        or training_receipt.get("best_macro_language_dev_loss") != 0.08414227871097675
        or training_receipt.get("selected_model_state_sha256")
        != plan.checkpoint.selected_state_sha256
        or training_receipt.get("calibrated") is not False
        or training_receipt.get("final_inference_performed") is not False
    ):
        raise V4CalibrationError("Training receipt is not the selected uncalibrated v4 checkpoint.")


def validate_v4_calibration_inputs(
    plan: Plan, ledger: Mapping[str, LicenseLedgerEntry]
) -> tuple[ManifestRow, ...]:
    """Validate the pair lock, its narrow ledger and v4 role isolation before CUDA is touched."""

    rows = tuple(load_manifest(plan.calibration.manifest.path))
    issues: list[str] = []
    try:
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
    except (ManifestError, V4CalibrationError) as error:
        issues.extend(error.issues)
    if len(rows) != plan.calibration.expected_rows:
        issues.append("Calibration row count differs from the immutable plan.")
    if Counter(row.label for row in rows) != Counter(plan.calibration.expected_labels.as_dict()):
        issues.append("Calibration label counts differ from the immutable plan.")
    if any(row.split != "test" or row.language != "ru" for row in rows):
        issues.append("Calibration pairs must remain RU assets with split='test'.")
    sources = tuple(sorted({row.source_name for row in rows}))
    if sources != plan.calibration.expected_sources:
        issues.append("Calibration source identities differ from the immutable plan.")
    for source in sources:
        entry = ledger.get(source)
        if entry is None or entry.train_dev_test_use != "research_only":
            issues.append(
                f"Calibration source {source!r} lacks a research-only fitting ledger entry."
            )
    pairs: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        pairs[row.text_id].append(row)
    if len(pairs) != plan.calibration.expected_pairs or any(
        len(pair) != 2
        or {row.label for row in pair} != {"bonafide", "spoof"}
        or len({row.text_hash for row in pair}) != 1
        for pair in pairs.values()
    ):
        issues.append(
            "Calibration manifest is not a complete one-bonafide/one-spoof exact pair lock."
        )
    try:
        train_rows = load_manifest(plan.evidence.train_manifest.path)
        dev_rows = load_manifest(plan.evidence.dev_manifest.path)
        validate_manifest([*train_rows, *dev_rows])
    except ManifestError as error:
        issues.extend(error.issues)
        train_rows = []
        dev_rows = []
    for role, other_rows in (("train", train_rows), ("dev", dev_rows)):
        for field in ("sample_id", "sha256", "text_hash", "parent_group_id"):
            overlap = {getattr(row, field) for row in rows}.intersection(
                getattr(row, field) for row in other_rows
            )
            if overlap:
                issues.append(f"Calibration/{role} overlap for {field}: {len(overlap)}.")
    if issues:
        raise V4CalibrationError(issues)
    return rows


def _cuda_device(plan: Plan) -> torch.device:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("v4 RU calibration requires an available CUDA BF16 device.")
    actual = Runtime(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_runtime=str(torch.version.cuda),
        transformers_version=transformers.__version__,
    )
    if actual != plan.runtime:
        raise RuntimeError(
            f"v4 calibration runtime lock mismatch: expected {plan.runtime!r}, got {actual!r}."
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
    """Create a receipt-safe serialization with all execution-relevant bindings."""

    return {
        "schema_version": V4_CALIBRATION_SCHEMA_VERSION,
        "protocol_id": V4_CALIBRATION_PROTOCOL_ID,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.path),
        "plan_sha256": plan.sha256,
        "protocol": plan.protocol,
        "license_ledger": _record(plan.license_ledger),
        "evidence": {
            "materialization_plan": _record(plan.evidence.materialization_plan),
            "materialization_receipt": _record(plan.evidence.materialization_receipt),
            "training_plan": _record(plan.evidence.training_plan),
            "training_receipt": _record(plan.evidence.training_receipt),
            "train_manifest": _record(plan.evidence.train_manifest),
            "dev_manifest": _record(plan.evidence.dev_manifest),
        },
        "checkpoint": {
            "path": str(plan.checkpoint.path),
            "sha256": plan.checkpoint.sha256,
            "selected_model_state_sha256": plan.checkpoint.selected_state_sha256,
        },
        "encoder": {
            "checkpoint_dir": str(plan.encoder.checkpoint_dir),
            "revision": plan.encoder.revision,
            "config": _record(plan.encoder.config),
            "weights": _record(plan.encoder.weights),
        },
        "head": asdict(plan.head),
        "calibration": {
            "manifest": _record(plan.calibration.manifest),
            "selected_split": "test",
            "expected_rows": plan.calibration.expected_rows,
            "expected_pairs": plan.calibration.expected_pairs,
            "expected_label_counts": plan.calibration.expected_labels.as_dict(),
            "expected_source_ids": list(plan.calibration.expected_sources),
        },
        "implementation": [_record(item) for item in plan.implementation],
        "runtime": asdict(plan.runtime),
        "inference": {**asdict(plan.inference), "device": "cuda", "precision": "bf16"},
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
            raise V4CalibrationError(f"Refusing to overwrite output: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preflight_v4_calibration(
    plan: Plan, rows: Sequence[ManifestRow], device: torch.device
) -> dict[str, object]:
    """Return the no-logit preflight record after all assets were byte-checked."""

    return {
        "status": "validated",
        "mode": "validate_only",
        "run_plan": plan_record(plan),
        "assets_validated": len(rows),
        "environment": _environment(device),
        "checkpoint_file_verified": True,
        "checkpoint_loaded": False,
        "calibration_scoring_performed": False,
        "temperature_fitted": False,
        "threshold_selection_performed": False,
        "final_inference_performed": False,
        "detector_feedback": False,
    }


def write_v4_calibration_preflight(plan: Plan, preflight: Mapping[str, object]) -> str:
    """Publish the required no-logit preflight exactly once and return its digest."""

    _write_exclusive_json(plan.outputs.preflight, preflight)
    return sha256_file(plan.outputs.preflight)


def _require_preflight(plan: Plan) -> None:
    receipt = _read_json(plan.outputs.preflight, "v4 calibration preflight receipt")
    run_plan = receipt.get("run_plan")
    if (
        receipt.get("status") != "validated"
        or receipt.get("mode") != "validate_only"
        or receipt.get("checkpoint_file_verified") is not True
        or receipt.get("checkpoint_loaded") is not False
        or receipt.get("calibration_scoring_performed") is not False
        or receipt.get("temperature_fitted") is not False
        or receipt.get("final_inference_performed") is not False
        or receipt.get("detector_feedback") is not False
        or not isinstance(run_plan, dict)
        or run_plan.get("protocol_id") != V4_CALIBRATION_PROTOCOL_ID
        or run_plan.get("plan_sha256") != plan.sha256
    ):
        raise V4CalibrationError("v4 calibration preflight does not bind this immutable plan.")


def validate_v4_calibration_checkpoint_file(plan: Plan) -> None:
    """Verify the ignored checkpoint bytes without deserializing or loading the model."""

    if (
        not plan.checkpoint.path.is_file()
        or sha256_file(plan.checkpoint.path) != plan.checkpoint.sha256
    ):
        raise V4CalibrationError("v4 selected checkpoint is missing or its file digest changed.")


def load_v4_calibration_state(plan: Plan) -> dict[str, Tensor]:
    """Read the ignored selected v4 checkpoint only after the preflight lock exists."""

    validate_v4_calibration_checkpoint_file(plan)
    with torch.serialization.safe_globals([TorchVersion]):
        value: object = torch.load(plan.checkpoint.path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise V4CalibrationError("v4 checkpoint root must be a dictionary.")
    checkpoint = cast(dict[str, object], value)
    if (
        checkpoint.get("model_name") != "xlsr_sls_model_v4"
        or checkpoint.get("training_purpose") != "research"
        or checkpoint.get("run_id") != "xlsr-sls-model-v4-train-v1"
        or checkpoint.get("selected_model_state_sha256") != plan.checkpoint.selected_state_sha256
    ):
        raise V4CalibrationError("v4 checkpoint identity does not match the calibration contract.")
    state_value = checkpoint.get("model_state_dict")
    if not isinstance(state_value, dict) or not state_value:
        raise V4CalibrationError("v4 checkpoint has no model_state_dict.")
    state = cast(dict[str, Tensor], state_value)
    if any(not isinstance(key, str) or not isinstance(item, Tensor) for key, item in state.items()):
        raise V4CalibrationError("v4 model_state_dict has invalid entries.")
    if state_dict_sha256(state) != plan.checkpoint.selected_state_sha256:
        raise V4CalibrationError(
            "v4 checkpoint state digest does not match the calibration contract."
        )
    return state


def build_v4_calibration_model(
    plan: Plan, state: Mapping[str, Tensor], device: torch.device
) -> XlsrSlsClassifier:
    """Construct the pinned XLS-R+SLS architecture and load the selected state strictly."""

    model = XlsrSlsClassifier.from_pretrained(
        str(plan.encoder.checkpoint_dir),
        attention_size=plan.head.attention_size,
        classifier_size=plan.head.classifier_size,
        dropout=plan.head.dropout,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise V4CalibrationError(
            "v4 checkpoint could not be loaded strictly into the pinned model."
        )
    return model.eval().to(device)


def infer_v4_calibration_logits(
    plan: Plan,
    rows: Sequence[ManifestRow],
    model: XlsrSlsClassifier,
    device: torch.device,
    audio_root: Path,
) -> tuple[Tensor, Tensor]:
    """Produce exactly one calibration logit per frozen asset in manifest order."""

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
        raise RuntimeError(
            "Calibration inference did not produce exactly one logit per frozen row."
        )
    return (
        torch.tensor([logits_by_id[row.sample_id] for row in rows]),
        torch.tensor([labels_by_id[row.sample_id] for row in rows]),
    )


def execute_v4_calibration(
    plan: Plan,
    rows: Sequence[ManifestRow],
    device: torch.device,
    audio_root: Path,
    preflight: Mapping[str, object],
) -> dict[str, object]:
    """Run the sole permitted scoring and temperature fit after a bound preflight."""

    _require_preflight(plan)
    existing = [
        str(path) for path in (plan.outputs.execution_lock, plan.outputs.report) if path.exists()
    ]
    if existing:
        raise V4CalibrationError(
            "Refusing repeated v4 calibration execution: " + ", ".join(existing)
        )
    execution_lock = {
        **preflight,
        "status": "calibration_scoring_started",
        "mode": "one_time_gpu_calibration",
        "preflight": {
            "path": str(plan.outputs.preflight),
            "sha256": sha256_file(plan.outputs.preflight),
        },
        "one_time_execution": True,
        "checkpoint_loaded": True,
        "calibration_scoring_performed": True,
        "temperature_fitted": False,
        "report_path": str(plan.outputs.report),
    }
    _write_exclusive_json(plan.outputs.execution_lock, execution_lock)
    torch.cuda.reset_peak_memory_stats(device)
    state = load_v4_calibration_state(plan)
    model = build_v4_calibration_model(plan, state, device)
    logits, labels = infer_v4_calibration_logits(plan, rows, model, device, audio_root)
    expected_labels = torch.tensor([1.0 if row.label == "spoof" else 0.0 for row in rows])
    if not torch.equal(labels, expected_labels):
        raise RuntimeError("Calibration labels differ from the frozen pair manifest.")
    calibration: CalibrationReport = TemperatureScaler().fit(
        logits, labels, max_iter=plan.inference.temperature_max_iter
    )
    torch.cuda.synchronize(device)
    report = {
        **preflight,
        "status": "ok",
        "mode": "one_time_gpu_calibration",
        "execution_lock": _record(
            PinnedFile(plan.outputs.execution_lock, sha256_file(plan.outputs.execution_lock))
        ),
        "frozen_checkpoint": {
            "path": str(plan.checkpoint.path),
            "sha256": plan.checkpoint.sha256,
            "selected_model_state_sha256": plan.checkpoint.selected_state_sha256,
        },
        "calibration": {
            **asdict(calibration),
            "records": len(rows),
            "pairs": plan.calibration.expected_pairs,
            "manifest": str(plan.calibration.manifest.path),
            "manifest_sha256": plan.calibration.manifest.sha256,
            "threshold_selection_performed": False,
        },
        "checkpoint_loaded": True,
        "calibration_scoring_performed": True,
        "temperature_fitted": True,
        "final_inference_performed": False,
        "detector_feedback": False,
        "limitations": [
            "This is personal research only, not product quality or a fraud-risk score.",
            "Temperature was fitted only on the frozen 73 RU VoxForge/eSpeak pairs.",
            "No threshold, epoch, model, architecture, augmentation or final asset was "
            "selected here.",
            "The scalar is RU-only; no calibrated KK probability claim is made.",
            "Speaker independence is not verified.",
        ],
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }
    _write_exclusive_json(plan.outputs.report, report)
    return report
