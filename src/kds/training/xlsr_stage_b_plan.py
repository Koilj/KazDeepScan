"""Strict plans for partial XLS-R fine-tuning initialized from a Stage-A head."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from kds.data.licenses import LicenseLedgerEntry
from kds.data.manifest import ManifestRow, load_manifest
from kds.training.xlsr_stage_a_plan import (
    PinnedFile,
    SelectedXlsrStageARows,
    XlsrStageAPlan,
    XlsrStageAPlanError,
    XlsrStageAProtocolReport,
    XlsrStageASelection,
    load_xlsr_stage_a_plan,
    validate_and_select_xlsr_stage_a,
    xlsr_stage_a_plan_record,
)

XLSR_STAGE_B_PLAN_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class XlsrStageBPlanError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class PinnedStageAHead:
    checkpoint: PinnedFile
    state_dict_sha256: str


@dataclass(frozen=True, slots=True)
class XlsrStageBTrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    window_samples: int
    sample_rate: int
    encoder_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    num_workers: int
    pin_memory: bool
    device: str
    precision: str
    last_encoder_blocks: int
    gradient_checkpointing: bool
    selection_metric: str


@dataclass(frozen=True, slots=True)
class XlsrStageBOutputs:
    checkpoint: Path
    report: Path


@dataclass(frozen=True, slots=True)
class XlsrStageBPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    base_plan_file: PinnedFile
    base_stage_a_plan: XlsrStageAPlan
    initial_head: PinnedStageAHead
    dev: XlsrStageASelection
    training: XlsrStageBTrainingConfig
    outputs: XlsrStageBOutputs


@dataclass(frozen=True, slots=True)
class XlsrStageBProtocolReport:
    run_id: str
    purpose: str
    train_dev: XlsrStageAProtocolReport
    initial_stage_a_dev_rows: int
    initial_stage_a_dev_overlap: dict[str, int]


def load_xlsr_stage_b_plan(path: Path) -> XlsrStageBPlan:
    if not path.is_file():
        raise XlsrStageBPlanError([f"XLS-R Stage-B plan does not exist: {path}"])
    try:
        plan_bytes = path.read_bytes()
        raw_value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsrStageBPlanError([f"Cannot read XLS-R Stage-B plan {path}: {error}"]) from error
    raw = _object(raw_value, "XLS-R Stage-B plan")
    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "purpose",
            "base_stage_a_plan",
            "initial_head",
            "dev",
            "training",
            "outputs",
        },
        "XLS-R Stage-B plan",
    )
    if raw["schema_version"] != XLSR_STAGE_B_PLAN_SCHEMA_VERSION:
        raise XlsrStageBPlanError(
            [
                "XLS-R Stage-B schema_version must be "
                f"{XLSR_STAGE_B_PLAN_SCHEMA_VERSION!r}, got {raw['schema_version']!r}."
            ]
        )
    if _required_string(raw, "purpose", "XLS-R Stage-B plan") != "research":
        raise XlsrStageBPlanError(["XLS-R Stage-B plans support purpose='research' only."])
    run_id = _required_string(raw, "run_id", "XLS-R Stage-B plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise XlsrStageBPlanError(
            ["run_id must use lowercase letters, digits, dots, underscores or hyphens."]
        )

    base_directory = path.resolve().parent
    base_plan_file = _parse_pinned_file(
        raw["base_stage_a_plan"], "base_stage_a_plan", base_directory
    )
    _verify_pinned_file(base_plan_file, "Base Stage-A plan")
    try:
        base_plan = load_xlsr_stage_a_plan(base_plan_file.path)
    except XlsrStageAPlanError as error:
        raise XlsrStageBPlanError(error.issues) from error
    initial_head = _parse_initial_head(raw["initial_head"], base_directory)
    _verify_pinned_file(initial_head.checkpoint, "Initial Stage-A head")
    if initial_head.checkpoint.path != base_plan.outputs.checkpoint:
        raise XlsrStageBPlanError(
            ["initial_head checkpoint must equal the base Stage-A checkpoint output path."]
        )
    dev = _parse_dev(raw["dev"], base_directory)
    _verify_pinned_file(dev.manifest, "Stage-B dev manifest")
    if dev.manifest.path == base_plan.dev.manifest.path:
        raise XlsrStageBPlanError(["Stage B requires a fresh dev manifest, not Stage-A dev."])
    training = _parse_training(raw["training"])
    outputs = _parse_outputs(raw["outputs"], base_directory)
    if outputs.checkpoint == outputs.report:
        raise XlsrStageBPlanError(["Checkpoint and report output paths must be different."])
    return XlsrStageBPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        base_plan_file=base_plan_file,
        base_stage_a_plan=base_plan,
        initial_head=initial_head,
        dev=dev,
        training=training,
        outputs=outputs,
    )


def validate_and_select_xlsr_stage_b(
    plan: XlsrStageBPlan, ledger: Mapping[str, LicenseLedgerEntry]
) -> tuple[XlsrStageBProtocolReport, SelectedXlsrStageARows]:
    """Validate the new train/dev protocol and exclude all Stage-A dev observations."""

    shadow_plan = replace(
        plan.base_stage_a_plan,
        run_id=plan.run_id,
        dev=plan.dev,
    )
    try:
        report, selected = validate_and_select_xlsr_stage_a(shadow_plan, ledger)
    except XlsrStageAPlanError as error:
        raise XlsrStageBPlanError(error.issues) from error
    old_dev_rows = [
        row
        for row in load_manifest(plan.base_stage_a_plan.dev.manifest.path)
        if row.split == plan.base_stage_a_plan.dev.source_split
    ]
    overlap = {
        "sample_id": _overlap_count(old_dev_rows, selected.dev, "sample_id"),
        "asset_sha256": _overlap_count(old_dev_rows, selected.dev, "sha256"),
        "text_hash": _overlap_count(old_dev_rows, selected.dev, "text_hash"),
        "parent_group_id": _overlap_count(old_dev_rows, selected.dev, "parent_group_id"),
    }
    nonzero = {name: count for name, count in overlap.items() if count}
    if nonzero:
        raise XlsrStageBPlanError(
            [
                "Fresh Stage-B dev overlaps the Stage-A selection dev: "
                + ", ".join(f"{name}={count}" for name, count in sorted(nonzero.items()))
                + "."
            ]
        )
    return (
        XlsrStageBProtocolReport(
            run_id=plan.run_id,
            purpose="research",
            train_dev=report,
            initial_stage_a_dev_rows=len(old_dev_rows),
            initial_stage_a_dev_overlap=overlap,
        ),
        selected,
    )


def xlsr_stage_b_plan_record(plan: XlsrStageBPlan) -> dict[str, object]:
    return {
        "schema_version": XLSR_STAGE_B_PLAN_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.plan_path),
        "plan_sha256": plan.plan_sha256,
        "base_stage_a_plan": {
            "path": str(plan.base_plan_file.path),
            "sha256": plan.base_plan_file.sha256,
            "record": xlsr_stage_a_plan_record(plan.base_stage_a_plan),
        },
        "initial_head": {
            "checkpoint": {
                "path": str(plan.initial_head.checkpoint.path),
                "sha256": plan.initial_head.checkpoint.sha256,
            },
            "state_dict_sha256": plan.initial_head.state_dict_sha256,
        },
        "dev": {
            "manifest": {
                "path": str(plan.dev.manifest.path),
                "sha256": plan.dev.manifest.sha256,
            },
            "source_split": plan.dev.source_split,
            "expected_source_ids": list(plan.dev.expected_source_ids),
            "expected_languages": list(plan.dev.expected_languages),
        },
        "training": asdict(plan.training),
        "outputs": {
            "checkpoint": str(plan.outputs.checkpoint),
            "report": str(plan.outputs.report),
        },
    }


def _overlap_count(
    old_rows: Iterable[ManifestRow], new_rows: Iterable[ManifestRow], field: str
) -> int:
    return len(
        {cast(str, getattr(row, field)) for row in old_rows}.intersection(
            cast(str, getattr(row, field)) for row in new_rows
        )
    )


def _parse_initial_head(value: object, base_directory: Path) -> PinnedStageAHead:
    raw = _object(value, "initial_head")
    _expect_exact_keys(raw, {"checkpoint", "state_dict_sha256"}, "initial_head")
    state_sha256 = _required_string(raw, "state_dict_sha256", "initial_head")
    if _SHA256.fullmatch(state_sha256) is None:
        raise XlsrStageBPlanError(
            ["initial_head state_dict_sha256 must contain 64 lowercase hexadecimal digits."]
        )
    return PinnedStageAHead(
        checkpoint=_parse_pinned_file(raw["checkpoint"], "initial_head checkpoint", base_directory),
        state_dict_sha256=state_sha256,
    )


def _parse_dev(value: object, base_directory: Path) -> XlsrStageASelection:
    raw = _object(value, "dev")
    _expect_exact_keys(
        raw,
        {"manifest", "source_split", "expected_source_ids", "expected_languages"},
        "dev",
    )
    source_split = _required_string(raw, "source_split", "dev")
    if source_split != "dev":
        raise XlsrStageBPlanError(["Stage-B dev source_split must be exactly 'dev'."])
    return XlsrStageASelection(
        manifest=_parse_pinned_file(raw["manifest"], "dev manifest", base_directory),
        source_split=source_split,
        expected_source_ids=_string_tuple(raw["expected_source_ids"], "dev sources"),
        expected_languages=_string_tuple(raw["expected_languages"], "dev languages"),
    )


def _parse_training(value: object) -> XlsrStageBTrainingConfig:
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
            "encoder_learning_rate",
            "head_learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "num_workers",
            "pin_memory",
            "device",
            "precision",
            "last_encoder_blocks",
            "gradient_checkpointing",
            "selection_metric",
        },
        "training",
    )
    device = _required_string(raw, "device", "training")
    precision = _required_string(raw, "precision", "training")
    checkpointing = _required_bool(raw, "gradient_checkpointing", "training")
    selection_metric = _required_string(raw, "selection_metric", "training")
    issues: list[str] = []
    if device != "cuda":
        issues.append("Stage-B device must be exactly 'cuda'.")
    if precision != "bf16":
        issues.append("Stage-B precision must be exactly 'bf16'.")
    if not checkpointing:
        issues.append("Stage-B requires gradient_checkpointing=true.")
    if selection_metric != "dev_loss":
        issues.append("Stage-B selection_metric must be exactly 'dev_loss'.")
    if issues:
        raise XlsrStageBPlanError(issues)
    return XlsrStageBTrainingConfig(
        seed=_required_int(raw, "seed", "training", minimum=0),
        epochs=_required_int(raw, "epochs", "training", minimum=1),
        batch_size=_required_int(raw, "batch_size", "training", minimum=1),
        gradient_accumulation_steps=_required_int(
            raw, "gradient_accumulation_steps", "training", minimum=1
        ),
        window_samples=_required_int(raw, "window_samples", "training", minimum=1),
        sample_rate=_required_int(raw, "sample_rate", "training", minimum=1),
        encoder_learning_rate=_required_float(
            raw, "encoder_learning_rate", "training", minimum=0.0, exclusive=True
        ),
        head_learning_rate=_required_float(
            raw, "head_learning_rate", "training", minimum=0.0, exclusive=True
        ),
        weight_decay=_required_float(raw, "weight_decay", "training", minimum=0.0),
        gradient_clip_norm=_required_float(
            raw, "gradient_clip_norm", "training", minimum=0.0, exclusive=True
        ),
        num_workers=_required_int(raw, "num_workers", "training", minimum=0),
        pin_memory=_required_bool(raw, "pin_memory", "training"),
        device=device,
        precision=precision,
        last_encoder_blocks=_required_int(raw, "last_encoder_blocks", "training", minimum=1),
        gradient_checkpointing=checkpointing,
        selection_metric=selection_metric,
    )


def _parse_outputs(value: object, base_directory: Path) -> XlsrStageBOutputs:
    raw = _object(value, "outputs")
    _expect_exact_keys(raw, {"checkpoint", "report"}, "outputs")
    return XlsrStageBOutputs(
        checkpoint=_relative_path(raw, "checkpoint", "outputs", base_directory),
        report=_relative_path(raw, "report", "outputs", base_directory),
    )


def _parse_pinned_file(value: object, label: str, base_directory: Path) -> PinnedFile:
    raw = _object(value, label)
    _expect_exact_keys(raw, {"path", "sha256"}, label)
    sha256 = _required_string(raw, "sha256", label)
    if _SHA256.fullmatch(sha256) is None:
        raise XlsrStageBPlanError([f"{label} sha256 must contain 64 lowercase hexadecimal digits."])
    return PinnedFile(
        path=_relative_path(raw, "path", label, base_directory),
        sha256=sha256,
    )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise XlsrStageBPlanError([f"{label} must be a non-empty JSON array."])
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise XlsrStageBPlanError([f"{label} values must be non-empty strings."])
    result = tuple(sorted(cast(str, item).strip() for item in value))
    if len(result) != len(set(result)):
        raise XlsrStageBPlanError([f"{label} must not contain duplicates."])
    return result


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise XlsrStageBPlanError([f"{label} must be a JSON object."])
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
        raise XlsrStageBPlanError(issues)


def _required_string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise XlsrStageBPlanError([f"{label} field {name!r} must be a non-empty string."])
    return value.strip()


def _required_int(raw: Mapping[str, object], name: str, label: str, *, minimum: int) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise XlsrStageBPlanError(
            [f"{label} field {name!r} must be an integer greater than or equal to {minimum}."]
        )
    return value


def _required_float(
    raw: Mapping[str, object],
    name: str,
    label: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise XlsrStageBPlanError([f"{label} field {name!r} must be a finite number."])
    result = float(value)
    below = result <= minimum if exclusive else result < minimum
    if not math.isfinite(result) or below:
        raise XlsrStageBPlanError([f"{label} field {name!r} is outside its valid range."])
    return result


def _required_bool(raw: Mapping[str, object], name: str, label: str) -> bool:
    value = raw.get(name)
    if not isinstance(value, bool):
        raise XlsrStageBPlanError([f"{label} field {name!r} must be boolean."])
    return value


def _relative_path(raw: Mapping[str, object], name: str, label: str, base_directory: Path) -> Path:
    value = Path(_required_string(raw, name, label))
    if value.is_absolute():
        raise XlsrStageBPlanError([f"{label} field {name!r} must be a relative path."])
    return (base_directory / value).resolve()


def _verify_pinned_file(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file():
        raise XlsrStageBPlanError([f"{label} does not exist: {pinned.path}"])
    digest = hashlib.sha256()
    try:
        with pinned.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise XlsrStageBPlanError(
            [f"Cannot hash {label.lower()} {pinned.path}: {error}"]
        ) from error
    actual = digest.hexdigest()
    if actual != pinned.sha256:
        raise XlsrStageBPlanError(
            [f"{label} SHA-256 mismatch for {pinned.path}: expected {pinned.sha256}, got {actual}."]
        )
