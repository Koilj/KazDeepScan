"""Immutable XLS-R+SLS model-v4 bilingual training contract primitives."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from torch.utils.data import Sampler

from kds.data.assets import sha256_file
from kds.data.augmentation import SymmetricTrainAugmentation
from kds.data.licenses import (
    APPROVED_LICENSE_STATUSES,
    LicenseLedgerEntry,
    LicenseLedgerError,
    validate_manifest_licenses,
)
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.training.xlsr_stage_a_plan import PinnedFile, PinnedXlsrEncoder

V4_TRAINING_PLAN_SCHEMA_VERSION = 1
V4_TRAINING_PROTOCOL_ID = "xlsr-sls-model-v4-training-v1"
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CELLS = ("kk/bonafide", "kk/spoof", "ru/bonafide", "ru/spoof")


class V4TrainingPlanError(ValueError):
    """Raised when a v4 training contract is not immutable or internally consistent."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class V4Selection:
    manifest: PinnedFile
    source_split: str
    expected_source_ids: tuple[str, ...]
    expected_cell_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class V4HeadConfig:
    attention_size: int
    classifier_size: int
    dropout: float


@dataclass(frozen=True, slots=True)
class V4TrainingConfig:
    seed: int
    warmup_epochs: int
    unfreeze_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    window_samples: int
    sample_rate: int
    head_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    last_encoder_blocks: int
    num_workers: int
    pin_memory: bool
    device: str
    precision: str
    selection_metric: str
    sampler: str
    augmentation: SymmetricTrainAugmentation


@dataclass(frozen=True, slots=True)
class V4RuntimeLock:
    python_version: str
    torch_version: str
    cuda_runtime: str
    transformers_version: str


@dataclass(frozen=True, slots=True)
class V4Outputs:
    checkpoint: Path
    report: Path
    execution_lock: Path


@dataclass(frozen=True, slots=True)
class V4TrainingPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    license_ledger: PinnedFile
    train: V4Selection
    dev: V4Selection
    encoder: PinnedXlsrEncoder
    head: V4HeadConfig
    training: V4TrainingConfig
    runtime: V4RuntimeLock
    implementation: tuple[PinnedFile, ...]
    outputs: V4Outputs


@dataclass(frozen=True, slots=True)
class V4RoleReport:
    role: str
    manifest_path: str
    rows: int
    cell_counts: dict[str, int]
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class V4ProtocolReport:
    run_id: str
    purpose: str
    roles: tuple[V4RoleReport, ...]
    train_dev_overlap: dict[str, int]
    no_calibration_or_final_inputs: bool


@dataclass(frozen=True, slots=True)
class SelectedV4Rows:
    train: tuple[ManifestRow, ...]
    dev_ru: tuple[ManifestRow, ...]
    dev_kk: tuple[ManifestRow, ...]


class V4BalancedCellBatchSampler(Sampler[list[int]]):
    """Yield fully balanced RU/KK × bona-fide/spoof batches without padding or reuse."""

    def __init__(self, rows: Sequence[ManifestRow], *, batch_size: int, seed: int) -> None:
        if batch_size <= 0 or batch_size % len(_CELLS):
            raise ValueError("v4 balanced sampler batch_size must be a positive multiple of four.")
        groups: dict[str, list[int]] = {cell: [] for cell in _CELLS}
        for index, row in enumerate(rows):
            cell = _cell(row)
            if cell not in groups:
                raise ValueError(f"v4 balanced sampler received unsupported cell: {cell!r}.")
            groups[cell].append(index)
        counts = {cell: len(indices) for cell, indices in groups.items()}
        if not all(counts.values()) or len(set(counts.values())) != 1:
            raise ValueError("v4 balanced sampler requires equal non-empty four-cell train inputs.")
        self._groups = {cell: tuple(indices) for cell, indices in groups.items()}
        self._per_cell = batch_size // len(_CELLS)
        if len(next(iter(self._groups.values()))) % self._per_cell:
            raise ValueError("v4 balanced sampler refuses partial or duplicate-padded batches.")
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("v4 balanced sampler epoch must not be negative.")
        self._epoch = epoch

    def __len__(self) -> int:
        return len(next(iter(self._groups.values()))) // self._per_cell

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(f"{self._seed}:v4-cell-balanced:{self._epoch}")
        shuffled = {cell: list(indices) for cell, indices in self._groups.items()}
        for indices in shuffled.values():
            generator.shuffle(indices)
        batches: list[list[int]] = []
        for offset in range(0, len(next(iter(shuffled.values()))), self._per_cell):
            batch = [
                index
                for cell in _CELLS
                for index in shuffled[cell][offset : offset + self._per_cell]
            ]
            generator.shuffle(batch)
            batches.append(batch)
        generator.shuffle(batches)
        yield from batches


def load_v4_training_plan(path: Path) -> V4TrainingPlan:
    """Read the strict v4 plan and verify every hash-pinned input and implementation file."""

    if not path.is_file():
        raise V4TrainingPlanError([f"v4 training plan does not exist: {path}"])
    try:
        plan_bytes = path.read_bytes()
        value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4TrainingPlanError([f"Cannot read v4 training plan {path}: {error}"]) from error
    raw = _object(value, "v4 training plan")
    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "run_id",
            "purpose",
            "license_ledger",
            "train",
            "dev",
            "encoder",
            "head",
            "training",
            "runtime",
            "implementation",
            "outputs",
            "prohibitions",
        },
        "v4 training plan",
    )
    if raw["schema_version"] != V4_TRAINING_PLAN_SCHEMA_VERSION:
        raise V4TrainingPlanError(["v4 training plan schema_version is unsupported."])
    if raw["protocol_id"] != V4_TRAINING_PROTOCOL_ID:
        raise V4TrainingPlanError(["v4 training plan protocol_id is invalid."])
    if _string(raw, "purpose", "v4 training plan") != "research":
        raise V4TrainingPlanError(["v4 training supports purpose='research' only."])
    run_id = _string(raw, "run_id", "v4 training plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise V4TrainingPlanError(["v4 training run_id has invalid characters."])
    _validate_prohibitions(raw["prohibitions"])

    base = path.resolve().parent
    ledger = _pinned_file(raw["license_ledger"], "license ledger", base)
    train = _selection(raw["train"], "train", base)
    dev = _selection(raw["dev"], "dev", base)
    encoder = _encoder(raw["encoder"], base)
    head = _head(raw["head"])
    training = _training(raw["training"])
    runtime = _runtime(raw["runtime"])
    implementation = _implementation(raw["implementation"], base)
    outputs = _outputs(raw["outputs"], base)
    output_paths = (outputs.checkpoint, outputs.report, outputs.execution_lock)
    if len(set(output_paths)) != len(output_paths):
        raise V4TrainingPlanError(["v4 training outputs must use distinct paths."])

    for pinned, label in (
        (ledger, "v4 license ledger"),
        (train.manifest, "v4 train manifest"),
        (dev.manifest, "v4 dev manifest"),
        (encoder.config, "XLS-R config"),
        (encoder.weights, "XLS-R weights"),
        *((item, "v4 implementation") for item in implementation),
    ):
        _verify_pinned_file(pinned, label)
    return V4TrainingPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        license_ledger=ledger,
        train=train,
        dev=dev,
        encoder=encoder,
        head=head,
        training=training,
        runtime=runtime,
        implementation=implementation,
        outputs=outputs,
    )


def validate_and_select_v4_training(
    plan: V4TrainingPlan, ledger: Mapping[str, LicenseLedgerEntry]
) -> tuple[V4ProtocolReport, SelectedV4Rows]:
    """Validate role, rights and complete train/dev isolation before a v4 execution."""

    issues: list[str] = []
    selected: dict[str, list[ManifestRow]] = {}
    reports: list[V4RoleReport] = []
    for role, selection in (("train", plan.train), ("dev", plan.dev)):
        try:
            rows = load_manifest(selection.manifest.path)
            validate_manifest(rows)
        except ManifestError as error:
            issues.extend(error.issues)
            continue
        chosen = [row for row in rows if row.split == selection.source_split]
        selected[role] = chosen
        if not chosen:
            issues.append(f"v4 role={role!r} has no selected rows.")
            continue
        source_ids = tuple(sorted({row.source_name for row in chosen}))
        if source_ids != selection.expected_source_ids:
            issues.append(
                f"v4 role={role!r} sources changed: "
                f"expected {list(selection.expected_source_ids)!r}, "
                f"got {list(source_ids)!r}."
            )
        counts = _cell_counts(chosen)
        if counts != selection.expected_cell_counts:
            issues.append(
                f"v4 role={role!r} cells changed: expected {selection.expected_cell_counts!r}, "
                f"got {counts!r}."
            )
        try:
            validate_manifest_licenses(chosen, ledger)
        except LicenseLedgerError as error:
            issues.extend(error.issues)
        for source_id in source_ids:
            entry = ledger.get(source_id)
            if entry is not None and entry.status in APPROVED_LICENSE_STATUSES:
                if entry.train_dev_test_use not in {"research_only", "product_allowed"}:
                    issues.append(
                        f"v4 source {source_id!r} is not permitted for research training."
                    )
        reports.append(
            V4RoleReport(
                role=role,
                manifest_path=str(selection.manifest.path),
                rows=len(chosen),
                cell_counts=counts,
                source_ids=source_ids,
            )
        )
    train_rows = selected.get("train", [])
    dev_rows = selected.get("dev", [])
    overlap = {
        "sample_id": _overlap(train_rows, dev_rows, "sample_id"),
        "asset_sha256": _overlap(train_rows, dev_rows, "sha256"),
        "text_hash": _overlap(train_rows, dev_rows, "text_hash"),
        "parent_group_id": _overlap(train_rows, dev_rows, "parent_group_id"),
    }
    if any(overlap.values()):
        issues.append(f"v4 train/dev overlap is non-zero: {overlap!r}.")
    shared_sources = sorted(
        {row.source_name for row in train_rows}.intersection(row.source_name for row in dev_rows)
    )
    if shared_sources:
        issues.append("v4 source lineage overlaps train/dev: " + ", ".join(shared_sources) + ".")
    try:
        validate_manifest([*train_rows, *dev_rows])
    except ManifestError as error:
        issues.extend(error.issues)
    dev_ru = [row for row in dev_rows if row.language == "ru"]
    dev_kk = [row for row in dev_rows if row.language == "kk"]
    if not dev_ru or not dev_kk:
        issues.append("v4 macro dev metric requires both isolated RU and KK cells.")
    if issues:
        raise V4TrainingPlanError(issues)
    return (
        V4ProtocolReport(
            run_id=plan.run_id,
            purpose="research",
            roles=tuple(reports),
            train_dev_overlap=overlap,
            no_calibration_or_final_inputs=True,
        ),
        SelectedV4Rows(train=tuple(train_rows), dev_ru=tuple(dev_ru), dev_kk=tuple(dev_kk)),
    )


def v4_training_plan_record(plan: V4TrainingPlan) -> dict[str, object]:
    """Serialize the complete immutable plan into a checkpoint/report-safe structure."""

    def file_record(item: PinnedFile) -> dict[str, str]:
        return {"path": str(item.path), "sha256": item.sha256}

    def selection_record(selection: V4Selection) -> dict[str, object]:
        return {
            "manifest": file_record(selection.manifest),
            "source_split": selection.source_split,
            "expected_source_ids": list(selection.expected_source_ids),
            "expected_cell_counts": selection.expected_cell_counts,
        }

    return {
        "schema_version": V4_TRAINING_PLAN_SCHEMA_VERSION,
        "protocol_id": V4_TRAINING_PROTOCOL_ID,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.plan_path),
        "plan_sha256": plan.plan_sha256,
        "license_ledger": file_record(plan.license_ledger),
        "train": selection_record(plan.train),
        "dev": selection_record(plan.dev),
        "encoder": {
            "checkpoint_dir": str(plan.encoder.checkpoint_dir),
            "revision": plan.encoder.revision,
            "config": file_record(plan.encoder.config),
            "weights": file_record(plan.encoder.weights),
        },
        "head": asdict(plan.head),
        "training": asdict(plan.training),
        "runtime": asdict(plan.runtime),
        "implementation": [file_record(item) for item in plan.implementation],
        "outputs": {
            "checkpoint": str(plan.outputs.checkpoint),
            "report": str(plan.outputs.report),
            "execution_lock": str(plan.outputs.execution_lock),
        },
    }


def _selection(value: object, role: str, base: Path) -> V4Selection:
    raw = _object(value, role)
    _expect_exact_keys(
        raw, {"manifest", "source_split", "expected_source_ids", "expected_cell_counts"}, role
    )
    if _string(raw, "source_split", role) != role:
        raise V4TrainingPlanError([f"v4 {role} source_split must be exactly {role!r}."])
    source_ids = _strings(raw["expected_source_ids"], f"v4 {role} sources")
    cells = _cell_counts_config(raw["expected_cell_counts"], f"v4 {role} cells")
    return V4Selection(
        manifest=_pinned_file(raw["manifest"], f"v4 {role} manifest", base),
        source_split=role,
        expected_source_ids=source_ids,
        expected_cell_counts=cells,
    )


def _encoder(value: object, base: Path) -> PinnedXlsrEncoder:
    raw = _object(value, "encoder")
    _expect_exact_keys(raw, {"checkpoint_dir", "revision", "config", "weights"}, "encoder")
    directory = _relative_path(raw["checkpoint_dir"], "encoder.checkpoint_dir", base)
    if not directory.is_dir():
        raise V4TrainingPlanError([f"v4 XLS-R checkpoint directory does not exist: {directory}"])
    config = _encoder_file(raw["config"], "encoder config", directory, "config.json")
    weights = _encoder_file(raw["weights"], "encoder weights", directory, "pytorch_model.bin")
    return PinnedXlsrEncoder(
        checkpoint_dir=directory,
        revision=_string(raw, "revision", "encoder"),
        config=config,
        weights=weights,
    )


def _encoder_file(value: object, label: str, directory: Path, name: str) -> PinnedFile:
    raw = _object(value, label)
    _expect_exact_keys(raw, {"filename", "sha256"}, label)
    if _string(raw, "filename", label) != name:
        raise V4TrainingPlanError([f"{label} filename must be exactly {name!r}."])
    return PinnedFile(path=(directory / name).resolve(), sha256=_sha(raw, "sha256", label))


def _head(value: object) -> V4HeadConfig:
    raw = _object(value, "head")
    _expect_exact_keys(raw, {"attention_size", "classifier_size", "dropout"}, "head")
    dropout = _number(raw, "dropout", "head", minimum=0.0)
    if dropout >= 1.0:
        raise V4TrainingPlanError(["head.dropout must be below 1."])
    return V4HeadConfig(
        attention_size=_integer(raw, "attention_size", "head", minimum=1),
        classifier_size=_integer(raw, "classifier_size", "head", minimum=1),
        dropout=dropout,
    )


def _training(value: object) -> V4TrainingConfig:
    raw = _object(value, "training")
    expected = {
        "seed",
        "warmup_epochs",
        "unfreeze_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "window_samples",
        "sample_rate",
        "head_learning_rate",
        "encoder_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "last_encoder_blocks",
        "num_workers",
        "pin_memory",
        "device",
        "precision",
        "selection_metric",
        "sampler",
        "augmentation",
    }
    _expect_exact_keys(raw, expected, "training")
    device = _string(raw, "device", "training")
    precision = _string(raw, "precision", "training")
    metric = _string(raw, "selection_metric", "training")
    sampler = _string(raw, "sampler", "training")
    if device != "cuda" or precision != "bf16":
        raise V4TrainingPlanError(["v4 training requires CUDA BF16."])
    if metric != "macro_language_dev_loss_ru_kk":
        raise V4TrainingPlanError(["v4 selection_metric must be macro_language_dev_loss_ru_kk."])
    if sampler != "balanced_language_label_without_padding":
        raise V4TrainingPlanError(["v4 sampler must be balanced_language_label_without_padding."])
    batch_size = _integer(raw, "batch_size", "training", minimum=4)
    if batch_size % 4:
        raise V4TrainingPlanError(["v4 batch_size must be divisible by four cells."])
    return V4TrainingConfig(
        seed=_integer(raw, "seed", "training", minimum=0),
        warmup_epochs=_integer(raw, "warmup_epochs", "training", minimum=1),
        unfreeze_epochs=_integer(raw, "unfreeze_epochs", "training", minimum=1),
        batch_size=batch_size,
        gradient_accumulation_steps=_integer(
            raw, "gradient_accumulation_steps", "training", minimum=1
        ),
        window_samples=_integer(raw, "window_samples", "training", minimum=1),
        sample_rate=_integer(raw, "sample_rate", "training", minimum=1),
        head_learning_rate=_number(raw, "head_learning_rate", "training", minimum=0.0, strict=True),
        encoder_learning_rate=_number(
            raw, "encoder_learning_rate", "training", minimum=0.0, strict=True
        ),
        weight_decay=_number(raw, "weight_decay", "training", minimum=0.0),
        gradient_clip_norm=_number(raw, "gradient_clip_norm", "training", minimum=0.0, strict=True),
        last_encoder_blocks=_integer(raw, "last_encoder_blocks", "training", minimum=1),
        num_workers=_integer(raw, "num_workers", "training", minimum=0),
        pin_memory=_boolean(raw, "pin_memory", "training"),
        device=device,
        precision=precision,
        selection_metric=metric,
        sampler=sampler,
        augmentation=_augmentation(raw["augmentation"]),
    )


def _augmentation(value: object) -> SymmetricTrainAugmentation:
    raw = _object(value, "v4 augmentation")
    _expect_exact_keys(
        raw,
        {
            "policy_id",
            "applies_to",
            "seed_namespace",
            "channel_gain_db",
            "codec_simulation",
            "replay_simulation",
        },
        "v4 augmentation",
    )
    if _string(raw, "applies_to", "v4 augmentation") != "train_only_both_labels":
        raise V4TrainingPlanError(["v4 augmentation must apply only to both train labels."])
    gain = _number_pair(raw["channel_gain_db"], "v4 augmentation channel gain")
    codec = _object(raw["codec_simulation"], "v4 augmentation codec")
    _expect_exact_keys(codec, {"resample_rates_hz", "quantization_bits"}, "v4 augmentation codec")
    rates_raw = codec["resample_rates_hz"]
    if not isinstance(rates_raw, list) or not rates_raw:
        raise V4TrainingPlanError(["v4 augmentation codec rates must be a non-empty list."])
    rates = tuple(
        _integer({"value": item}, "value", "v4 codec rate", minimum=1) for item in rates_raw
    )
    replay = _object(raw["replay_simulation"], "v4 augmentation replay")
    _expect_exact_keys(replay, {"delay_ms", "attenuation"}, "v4 augmentation replay")
    delay = _number_pair(replay["delay_ms"], "v4 augmentation replay delay")
    attenuation = _number_pair(replay["attenuation"], "v4 augmentation replay attenuation")
    return SymmetricTrainAugmentation(
        policy_id=_string(raw, "policy_id", "v4 augmentation"),
        seed_namespace=_string(raw, "seed_namespace", "v4 augmentation"),
        channel_gain_db_min=gain[0],
        channel_gain_db_max=gain[1],
        codec_resample_rates_hz=rates,
        codec_quantization_bits=_integer(
            codec, "quantization_bits", "v4 augmentation codec", minimum=2
        ),
        replay_delay_ms_min=delay[0],
        replay_delay_ms_max=delay[1],
        replay_attenuation_min=attenuation[0],
        replay_attenuation_max=attenuation[1],
    )


def _runtime(value: object) -> V4RuntimeLock:
    raw = _object(value, "runtime")
    _expect_exact_keys(
        raw, {"python_version", "torch_version", "cuda_runtime", "transformers_version"}, "runtime"
    )
    return V4RuntimeLock(
        python_version=_string(raw, "python_version", "runtime"),
        torch_version=_string(raw, "torch_version", "runtime"),
        cuda_runtime=_string(raw, "cuda_runtime", "runtime"),
        transformers_version=_string(raw, "transformers_version", "runtime"),
    )


def _implementation(value: object, base: Path) -> tuple[PinnedFile, ...]:
    if not isinstance(value, list) or not value:
        raise V4TrainingPlanError(["v4 implementation must be a non-empty array."])
    files = tuple(_pinned_file(item, "v4 implementation", base) for item in value)
    if len({item.path for item in files}) != len(files):
        raise V4TrainingPlanError(["v4 implementation paths must be unique."])
    return files


def _outputs(value: object, base: Path) -> V4Outputs:
    raw = _object(value, "outputs")
    _expect_exact_keys(raw, {"checkpoint", "report", "execution_lock"}, "outputs")
    return V4Outputs(
        checkpoint=_relative_path(raw["checkpoint"], "outputs.checkpoint", base),
        report=_relative_path(raw["report"], "outputs.report", base),
        execution_lock=_relative_path(raw["execution_lock"], "outputs.execution_lock", base),
    )


def _validate_prohibitions(value: object) -> None:
    raw = _object(value, "prohibitions")
    expected = {
        "calibration",
        "final_evaluation",
        "final_paths",
        "detector_feedback",
        "checkpoint_overwrite",
        "execution_repeat",
        "network_downloads",
    }
    _expect_exact_keys(raw, expected, "prohibitions")
    if any(raw[name] is not True for name in expected):
        raise V4TrainingPlanError(["v4 prohibitions must all be true."])


def _pinned_file(value: object, label: str, base: Path) -> PinnedFile:
    raw = _object(value, label)
    _expect_exact_keys(raw, {"path", "sha256"}, label)
    return PinnedFile(
        path=_relative_path(raw["path"], f"{label}.path", base), sha256=_sha(raw, "sha256", label)
    )


def _verify_pinned_file(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file():
        raise V4TrainingPlanError([f"{label} is missing: {pinned.path}"])
    actual = sha256_file(pinned.path)
    if actual != pinned.sha256:
        raise V4TrainingPlanError(
            [f"{label} SHA-256 mismatch for {pinned.path}: expected {pinned.sha256}, got {actual}."]
        )


def _relative_path(value: object, label: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise V4TrainingPlanError([f"{label} must be a non-empty project-relative path."])
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or "\\" in value:
        raise V4TrainingPlanError([f"{label} is not a safe project-relative path."])
    candidate = (base / parsed).resolve()
    project = Path.cwd().resolve()
    try:
        candidate.relative_to(project)
    except ValueError as error:
        raise V4TrainingPlanError([f"{label} resolves outside the project root."]) from error
    return candidate


def _cell_counts(rows: Iterable[ManifestRow]) -> dict[str, int]:
    counts = Counter(_cell(row) for row in rows)
    return {cell: counts[cell] for cell in _CELLS if counts[cell]}


def _cell(row: ManifestRow) -> str:
    return f"{row.language}/{row.label}"


def _cell_counts_config(value: object, label: str) -> dict[str, int]:
    raw = _object(value, label)
    if set(raw) != set(_CELLS):
        raise V4TrainingPlanError([f"{label} must contain exactly {_CELLS!r}."])
    return {cell: _integer(raw, cell, label, minimum=1) for cell in _CELLS}


def _overlap(first: Sequence[ManifestRow], second: Sequence[ManifestRow], attribute: str) -> int:
    return len(
        {getattr(row, attribute) for row in first}.intersection(
            getattr(row, attribute) for row in second
        )
    )


def _number_pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise V4TrainingPlanError([f"{label} must contain exactly two finite numbers."])
    parsed = tuple(_number({"value": item}, "value", label, minimum=-math.inf) for item in value)
    if parsed[0] > parsed[1]:
        raise V4TrainingPlanError([f"{label} lower bound exceeds upper bound."])
    return cast(tuple[float, float], parsed)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4TrainingPlanError([f"{label} must be a JSON object."])
    return cast(dict[str, object], value)


def _expect_exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    issues: list[str] = []
    if missing:
        issues.append(f"{label} missing fields: {', '.join(missing)}.")
    if unknown:
        issues.append(f"{label} has unknown fields: {', '.join(unknown)}.")
    if issues:
        raise V4TrainingPlanError(issues)


def _string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise V4TrainingPlanError([f"{label}.{name} must be a non-empty string."])
    return value.strip()


def _strings(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise V4TrainingPlanError([f"{label} must be a non-empty string array."])
    result = tuple(sorted(cast(str, item) for item in value))
    if len(result) != len(set(result)):
        raise V4TrainingPlanError([f"{label} must not repeat source IDs."])
    return result


def _sha(raw: Mapping[str, object], name: str, label: str) -> str:
    value = _string(raw, name, label)
    if _SHA256.fullmatch(value) is None:
        raise V4TrainingPlanError([f"{label}.{name} must be a lowercase SHA-256 digest."])
    return value


def _integer(raw: Mapping[str, object], name: str, label: str, *, minimum: int) -> int:
    value = raw.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise V4TrainingPlanError([f"{label}.{name} must be an integer >= {minimum}."])
    return value


def _number(
    raw: Mapping[str, object], name: str, label: str, *, minimum: float, strict: bool = False
) -> float:
    value = raw.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise V4TrainingPlanError([f"{label}.{name} must be finite number."])
    result = float(value)
    if result < minimum or (strict and result <= minimum):
        comparator = ">" if strict else ">="
        raise V4TrainingPlanError([f"{label}.{name} must be {comparator} {minimum}."])
    return result


def _boolean(raw: Mapping[str, object], name: str, label: str) -> bool:
    value = raw.get(name)
    if not isinstance(value, bool):
        raise V4TrainingPlanError([f"{label}.{name} must be boolean."])
    return value
