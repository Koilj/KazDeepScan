"""Strict, hash-pinned plans for XLS-R+SLS Stage-A train/dev runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from kds.data.licenses import (
    APPROVED_LICENSE_STATUSES,
    LicenseLedgerEntry,
    LicenseLedgerError,
    validate_manifest_licenses,
)
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest

XLSR_STAGE_A_PLAN_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class XlsrStageAPlanError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class PinnedFile:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class XlsrStageASelection:
    manifest: PinnedFile
    source_split: str
    expected_source_ids: tuple[str, ...]
    expected_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PinnedXlsrEncoder:
    checkpoint_dir: Path
    revision: str
    config: PinnedFile
    weights: PinnedFile


@dataclass(frozen=True, slots=True)
class XlsrStageAHeadConfig:
    attention_size: int
    classifier_size: int
    dropout: float


@dataclass(frozen=True, slots=True)
class XlsrStageATrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    window_samples: int
    sample_rate: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    num_workers: int
    pin_memory: bool
    device: str
    precision: str
    freeze_encoder: bool
    encoder_eval_mode: bool
    selection_metric: str


@dataclass(frozen=True, slots=True)
class XlsrStageAOutputs:
    checkpoint: Path
    report: Path


@dataclass(frozen=True, slots=True)
class XlsrStageAPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    license_ledger: PinnedFile
    train: XlsrStageASelection
    dev: XlsrStageASelection
    encoder: PinnedXlsrEncoder
    head: XlsrStageAHeadConfig
    training: XlsrStageATrainingConfig
    outputs: XlsrStageAOutputs


@dataclass(frozen=True, slots=True)
class XlsrStageARoleReport:
    role: str
    manifest_path: str
    source_split: str
    rows: int
    label_counts: dict[str, int]
    source_ids: tuple[str, ...]
    languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class XlsrStageAProtocolReport:
    run_id: str
    purpose: str
    roles: tuple[XlsrStageARoleReport, ...]


@dataclass(frozen=True, slots=True)
class SelectedXlsrStageARows:
    train: tuple[ManifestRow, ...]
    dev: tuple[ManifestRow, ...]


def load_xlsr_stage_a_plan(path: Path) -> XlsrStageAPlan:
    """Load a strict plan and verify every file used by the Stage-A run."""

    if not path.is_file():
        raise XlsrStageAPlanError([f"XLS-R Stage-A plan does not exist: {path}"])
    try:
        plan_bytes = path.read_bytes()
        raw_value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsrStageAPlanError([f"Cannot read XLS-R Stage-A plan {path}: {error}"]) from error
    raw = _object(raw_value, "XLS-R Stage-A plan")
    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "purpose",
            "license_ledger",
            "train",
            "dev",
            "encoder",
            "head",
            "training",
            "outputs",
        },
        "XLS-R Stage-A plan",
    )
    if raw["schema_version"] != XLSR_STAGE_A_PLAN_SCHEMA_VERSION:
        raise XlsrStageAPlanError(
            [
                "XLS-R Stage-A schema_version must be "
                f"{XLSR_STAGE_A_PLAN_SCHEMA_VERSION!r}, got {raw['schema_version']!r}."
            ]
        )
    if _required_string(raw, "purpose", "XLS-R Stage-A plan") != "research":
        raise XlsrStageAPlanError(["XLS-R Stage-A plans support purpose='research' only."])
    run_id = _required_string(raw, "run_id", "XLS-R Stage-A plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise XlsrStageAPlanError(
            ["run_id must use lowercase letters, digits, dots, underscores or hyphens."]
        )

    base_directory = path.resolve().parent
    license_ledger = _parse_pinned_file(raw["license_ledger"], "license_ledger", base_directory)
    train = _parse_selection(raw["train"], "train", base_directory)
    dev = _parse_selection(raw["dev"], "dev", base_directory)
    encoder = _parse_encoder(raw["encoder"], base_directory)
    head = _parse_head(raw["head"])
    training = _parse_training(raw["training"])
    outputs = _parse_outputs(raw["outputs"], base_directory)
    if outputs.checkpoint == outputs.report:
        raise XlsrStageAPlanError(["Checkpoint and report output paths must be different."])

    for pinned, label in (
        (license_ledger, "License ledger"),
        (train.manifest, "Train manifest"),
        (dev.manifest, "Dev manifest"),
        (encoder.config, "XLS-R config"),
        (encoder.weights, "XLS-R weights"),
    ):
        _verify_pinned_file(pinned, label)
    return XlsrStageAPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        license_ledger=license_ledger,
        train=train,
        dev=dev,
        encoder=encoder,
        head=head,
        training=training,
        outputs=outputs,
    )


def validate_and_select_xlsr_stage_a(
    plan: XlsrStageAPlan, ledger: Mapping[str, LicenseLedgerEntry]
) -> tuple[XlsrStageAProtocolReport, SelectedXlsrStageARows]:
    """Validate rights and train/dev isolation, returning only the selected rows."""

    issues: list[str] = []
    selections: dict[str, list[ManifestRow]] = {}
    reports: list[XlsrStageARoleReport] = []
    for role, selection in (("train", plan.train), ("dev", plan.dev)):
        try:
            full_rows = load_manifest(selection.manifest.path)
            validate_manifest(full_rows)
        except ManifestError as error:
            issues.extend(error.issues)
            continue
        rows = [row for row in full_rows if row.split == selection.source_split]
        selections[role] = rows
        if not rows:
            issues.append(
                f"Stage-A role={role!r} selects no rows with split={selection.source_split!r}."
            )
            continue
        source_ids = tuple(sorted({row.source_name for row in rows}))
        languages = tuple(sorted({row.language for row in rows}))
        if source_ids != selection.expected_source_ids:
            issues.append(
                f"Stage-A role={role!r} expected sources "
                f"{list(selection.expected_source_ids)!r}, found {list(source_ids)!r}."
            )
        if languages != selection.expected_languages:
            issues.append(
                f"Stage-A role={role!r} expected languages "
                f"{list(selection.expected_languages)!r}, found {list(languages)!r}."
            )
        if {row.label for row in rows} != {"bonafide", "spoof"}:
            issues.append(f"Stage-A role={role!r} must include both bonafide and spoof rows.")
        try:
            validate_manifest_licenses(rows, ledger)
        except LicenseLedgerError as error:
            issues.extend(error.issues)
        for source_id in source_ids:
            entry = ledger.get(source_id)
            if entry is None or entry.status not in APPROVED_LICENSE_STATUSES:
                continue
            if entry.train_dev_test_use not in {"research_only", "product_allowed"}:
                issues.append(
                    f"Source {source_id!r} is prohibited for train_dev_test_use "
                    "in a research Stage-A run."
                )
        reports.append(
            XlsrStageARoleReport(
                role=role,
                manifest_path=str(selection.manifest.path),
                source_split=selection.source_split,
                rows=len(rows),
                label_counts={
                    label: sum(row.label == label for row in rows)
                    for label in ("bonafide", "spoof")
                },
                source_ids=source_ids,
                languages=languages,
            )
        )

    train_rows = selections.get("train", [])
    dev_rows = selections.get("dev", [])
    shared_sources = sorted(
        {row.source_name for row in train_rows}.intersection(row.source_name for row in dev_rows)
    )
    if shared_sources:
        issues.append("Source leakage between train/dev: " + ", ".join(shared_sources) + ".")
    try:
        validate_manifest([*train_rows, *dev_rows])
    except ManifestError as error:
        issues.extend(error.issues)
    if issues:
        raise XlsrStageAPlanError(issues)
    return (
        XlsrStageAProtocolReport(
            run_id=plan.run_id,
            purpose="research",
            roles=tuple(reports),
        ),
        SelectedXlsrStageARows(train=tuple(train_rows), dev=tuple(dev_rows)),
    )


def xlsr_stage_a_plan_record(plan: XlsrStageAPlan) -> dict[str, object]:
    """Return the immutable plan fields in a JSON/checkpoint-safe representation."""

    def pinned_record(pinned: PinnedFile) -> dict[str, str]:
        return {"path": str(pinned.path), "sha256": pinned.sha256}

    def selection_record(selection: XlsrStageASelection) -> dict[str, object]:
        return {
            "manifest": pinned_record(selection.manifest),
            "source_split": selection.source_split,
            "expected_source_ids": list(selection.expected_source_ids),
            "expected_languages": list(selection.expected_languages),
        }

    return {
        "schema_version": XLSR_STAGE_A_PLAN_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.plan_path),
        "plan_sha256": plan.plan_sha256,
        "license_ledger": pinned_record(plan.license_ledger),
        "train": selection_record(plan.train),
        "dev": selection_record(plan.dev),
        "encoder": {
            "checkpoint_dir": str(plan.encoder.checkpoint_dir),
            "revision": plan.encoder.revision,
            "config": pinned_record(plan.encoder.config),
            "weights": pinned_record(plan.encoder.weights),
        },
        "head": asdict(plan.head),
        "training": asdict(plan.training),
        "outputs": {
            "checkpoint": str(plan.outputs.checkpoint),
            "report": str(plan.outputs.report),
        },
    }


def _parse_selection(value: object, role: str, base_directory: Path) -> XlsrStageASelection:
    raw = _object(value, role)
    _expect_exact_keys(
        raw,
        {"manifest", "source_split", "expected_source_ids", "expected_languages"},
        role,
    )
    source_split = _required_string(raw, "source_split", role)
    if source_split != role:
        raise XlsrStageAPlanError(
            [f"Stage-A {role} source_split must be exactly {role!r}, got {source_split!r}."]
        )
    return XlsrStageASelection(
        manifest=_parse_pinned_file(raw["manifest"], f"{role} manifest", base_directory),
        source_split=source_split,
        expected_source_ids=_string_tuple(raw["expected_source_ids"], f"{role} sources"),
        expected_languages=_string_tuple(raw["expected_languages"], f"{role} languages"),
    )


def _parse_encoder(value: object, base_directory: Path) -> PinnedXlsrEncoder:
    raw = _object(value, "encoder")
    _expect_exact_keys(raw, {"checkpoint_dir", "revision", "config", "weights"}, "encoder")
    checkpoint_dir = _relative_path(raw, "checkpoint_dir", "encoder", base_directory)
    if not checkpoint_dir.is_dir():
        raise XlsrStageAPlanError([f"XLS-R checkpoint directory does not exist: {checkpoint_dir}"])
    config = _parse_encoder_file(raw["config"], "encoder config", checkpoint_dir)
    weights = _parse_encoder_file(raw["weights"], "encoder weights", checkpoint_dir)
    if config.path.name != "config.json":
        raise XlsrStageAPlanError(["Encoder config filename must be exactly 'config.json'."])
    if weights.path.name != "pytorch_model.bin":
        raise XlsrStageAPlanError(["Encoder weights filename must be exactly 'pytorch_model.bin'."])
    return PinnedXlsrEncoder(
        checkpoint_dir=checkpoint_dir,
        revision=_required_string(raw, "revision", "encoder"),
        config=config,
        weights=weights,
    )


def _parse_encoder_file(value: object, label: str, checkpoint_dir: Path) -> PinnedFile:
    raw = _object(value, label)
    _expect_exact_keys(raw, {"filename", "sha256"}, label)
    filename = _required_string(raw, "filename", label)
    if Path(filename).name != filename:
        raise XlsrStageAPlanError([f"{label} filename must be a plain filename."])
    return PinnedFile(path=(checkpoint_dir / filename).resolve(), sha256=_sha(raw, label))


def _parse_head(value: object) -> XlsrStageAHeadConfig:
    raw = _object(value, "head")
    _expect_exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    return XlsrStageAHeadConfig(
        attention_size=_required_int(raw, "attention_size", "head", minimum=1),
        classifier_size=_required_int(raw, "classifier_size", "head", minimum=1),
        dropout=_required_float(raw, "dropout", "head", minimum=0.0, maximum=1.0),
    )


def _parse_training(value: object) -> XlsrStageATrainingConfig:
    raw = _object(value, "training")
    _expect_exact_keys(
        raw,
        {
            "seed",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "window_samples",
            "sample_rate",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "num_workers",
            "pin_memory",
            "device",
            "precision",
            "freeze_encoder",
            "encoder_eval_mode",
            "selection_metric",
        },
        "training",
    )
    device = _required_string(raw, "device", "training")
    precision = _required_string(raw, "precision", "training")
    freeze_encoder = _required_bool(raw, "freeze_encoder", "training")
    encoder_eval_mode = _required_bool(raw, "encoder_eval_mode", "training")
    selection_metric = _required_string(raw, "selection_metric", "training")
    issues: list[str] = []
    if device != "cuda":
        issues.append("Stage-A device must be exactly 'cuda'.")
    if precision != "bf16":
        issues.append("Stage-A precision must be exactly 'bf16'.")
    if not freeze_encoder:
        issues.append("Stage-A requires freeze_encoder=true.")
    if not encoder_eval_mode:
        issues.append("Stage-A requires encoder_eval_mode=true.")
    if selection_metric != "dev_loss":
        issues.append("Stage-A selection_metric must be exactly 'dev_loss'.")
    if issues:
        raise XlsrStageAPlanError(issues)
    return XlsrStageATrainingConfig(
        seed=_required_int(raw, "seed", "training", minimum=0),
        epochs=_required_int(raw, "epochs", "training", minimum=1),
        batch_size=_required_int(raw, "batch_size", "training", minimum=1),
        gradient_accumulation_steps=_required_int(
            raw, "gradient_accumulation_steps", "training", minimum=1
        ),
        window_samples=_required_int(raw, "window_samples", "training", minimum=1),
        sample_rate=_required_int(raw, "sample_rate", "training", minimum=1),
        learning_rate=_required_float(
            raw, "learning_rate", "training", minimum=0.0, minimum_inclusive=False
        ),
        weight_decay=_required_float(raw, "weight_decay", "training", minimum=0.0),
        gradient_clip_norm=_required_float(
            raw, "gradient_clip_norm", "training", minimum=0.0, minimum_inclusive=False
        ),
        num_workers=_required_int(raw, "num_workers", "training", minimum=0),
        pin_memory=_required_bool(raw, "pin_memory", "training"),
        device=device,
        precision=precision,
        freeze_encoder=freeze_encoder,
        encoder_eval_mode=encoder_eval_mode,
        selection_metric=selection_metric,
    )


def _parse_outputs(value: object, base_directory: Path) -> XlsrStageAOutputs:
    raw = _object(value, "outputs")
    _expect_exact_keys(raw, {"checkpoint", "report"}, "outputs")
    return XlsrStageAOutputs(
        checkpoint=_relative_path(raw, "checkpoint", "outputs", base_directory),
        report=_relative_path(raw, "report", "outputs", base_directory),
    )


def _parse_pinned_file(value: object, label: str, base_directory: Path) -> PinnedFile:
    raw = _object(value, label)
    _expect_exact_keys(raw, {"path", "sha256"}, label)
    return PinnedFile(
        path=_relative_path(raw, "path", label, base_directory),
        sha256=_sha(raw, label),
    )


def _sha(raw: Mapping[str, object], label: str) -> str:
    sha256 = _required_string(raw, "sha256", label)
    if _SHA256.fullmatch(sha256) is None:
        raise XlsrStageAPlanError([f"{label} sha256 must contain 64 lowercase hexadecimal digits."])
    return sha256


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise XlsrStageAPlanError([f"{label} must be a non-empty JSON array."])
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise XlsrStageAPlanError([f"{label} values must be non-empty strings."])
    result = tuple(sorted(cast(str, item).strip() for item in value))
    if len(result) != len(set(result)):
        raise XlsrStageAPlanError([f"{label} must not contain duplicates."])
    return result


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise XlsrStageAPlanError([f"{label} must be a JSON object."])
    return cast(dict[str, object], value)


def _expect_exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    unknown = sorted(set(raw).difference(expected))
    missing = sorted(expected.difference(raw))
    issues: list[str] = []
    if missing:
        issues.append(f"{label} missing fields: " + ", ".join(missing) + ".")
    if unknown:
        issues.append(f"{label} has unknown fields: " + ", ".join(unknown) + ".")
    if issues:
        raise XlsrStageAPlanError(issues)


def _required_string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise XlsrStageAPlanError([f"{label} field {name!r} must be a non-empty string."])
    return value.strip()


def _required_int(raw: Mapping[str, object], name: str, label: str, *, minimum: int) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise XlsrStageAPlanError(
            [f"{label} field {name!r} must be an integer greater than or equal to {minimum}."]
        )
    return value


def _required_float(
    raw: Mapping[str, object],
    name: str,
    label: str,
    *,
    minimum: float,
    minimum_inclusive: bool = True,
    maximum: float | None = None,
) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise XlsrStageAPlanError([f"{label} field {name!r} must be a finite number."])
    result = float(value)
    below_minimum = result < minimum if minimum_inclusive else result <= minimum
    above_maximum = maximum is not None and result >= maximum
    if not math.isfinite(result) or below_minimum or above_maximum:
        raise XlsrStageAPlanError([f"{label} field {name!r} is outside its valid range."])
    return result


def _required_bool(raw: Mapping[str, object], name: str, label: str) -> bool:
    value = raw.get(name)
    if not isinstance(value, bool):
        raise XlsrStageAPlanError([f"{label} field {name!r} must be boolean."])
    return value


def _relative_path(raw: Mapping[str, object], name: str, label: str, base_directory: Path) -> Path:
    value = Path(_required_string(raw, name, label))
    if value.is_absolute():
        raise XlsrStageAPlanError([f"{label} field {name!r} must be a relative path."])
    return (base_directory / value).resolve()


def _verify_pinned_file(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file():
        raise XlsrStageAPlanError([f"{label} does not exist: {pinned.path}"])
    digest = hashlib.sha256()
    try:
        with pinned.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise XlsrStageAPlanError(
            [f"Cannot hash {label.lower()} {pinned.path}: {error}"]
        ) from error
    actual = digest.hexdigest()
    if actual != pinned.sha256:
        raise XlsrStageAPlanError(
            [f"{label} SHA-256 mismatch for {pinned.path}: expected {pinned.sha256}, got {actual}."]
        )
