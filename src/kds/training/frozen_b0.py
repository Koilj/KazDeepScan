"""Strict pre-registered run plans for frozen B0 unseen-generator evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from torch import Tensor

from kds.data.unseen_generator_ood import UnseenGeneratorSuite
from kds.models import B0Config
from kds.training.b0 import EpochResult

FROZEN_B0_RUN_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FrozenB0RunPlanError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class FrozenB0TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    window_samples: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str


@dataclass(frozen=True, slots=True)
class FrozenB0Outputs:
    checkpoint: Path
    report: Path


@dataclass(frozen=True, slots=True)
class FrozenInputFile:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class FrozenB0RunPlan:
    run_id: str
    plan_path: Path
    plan_sha256: str
    suite_path: Path
    suite_sha256: str
    license_ledger: FrozenInputFile
    manifests: tuple[FrozenInputFile, ...]
    model_config: B0Config
    training: FrozenB0TrainingConfig
    outputs: FrozenB0Outputs


def load_frozen_b0_run_plan(path: Path) -> FrozenB0RunPlan:
    """Load a strict run plan and verify that its frozen suite has not changed."""

    if not path.is_file():
        raise FrozenB0RunPlanError([f"Frozen B0 run plan does not exist: {path}"])
    try:
        plan_bytes = path.read_bytes()
        raw_value: object = json.loads(plan_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenB0RunPlanError([f"Cannot read frozen B0 run plan {path}: {error}"]) from error
    if not isinstance(raw_value, dict):
        raise FrozenB0RunPlanError(["Frozen B0 run plan root must be a JSON object."])
    raw = cast(dict[str, object], raw_value)
    _expect_exact_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "purpose",
            "suite",
            "license_ledger",
            "manifests",
            "model",
            "training",
            "outputs",
        },
        "Frozen B0 run plan",
    )
    if raw["schema_version"] != FROZEN_B0_RUN_SCHEMA_VERSION:
        raise FrozenB0RunPlanError(
            [
                "Frozen B0 run plan schema_version must be "
                f"{FROZEN_B0_RUN_SCHEMA_VERSION!r}, got {raw['schema_version']!r}."
            ]
        )
    if _required_string(raw, "purpose", "Frozen B0 run plan") != "research":
        raise FrozenB0RunPlanError(["Frozen B0 run plans support purpose='research' only."])
    run_id = _required_string(raw, "run_id", "Frozen B0 run plan")
    if _RUN_ID.fullmatch(run_id) is None:
        raise FrozenB0RunPlanError(
            ["run_id must use lowercase letters, digits, dots, underscores or hyphens."]
        )

    base_directory = path.resolve().parent
    suite_path, suite_sha256 = _parse_suite(raw["suite"], base_directory)
    license_ledger = _parse_input_file(raw["license_ledger"], "license_ledger", base_directory)
    manifests = _parse_manifests(raw["manifests"], base_directory)
    model_config = _parse_model(raw["model"])
    training = _parse_training(raw["training"])
    outputs = _parse_outputs(raw["outputs"], base_directory)
    if outputs.checkpoint == outputs.report:
        raise FrozenB0RunPlanError(["Checkpoint and report output paths must be different."])

    actual_suite_sha256 = _file_sha256(suite_path, "Frozen unseen-generator suite")
    if actual_suite_sha256 != suite_sha256:
        raise FrozenB0RunPlanError(
            [
                f"Frozen suite SHA-256 mismatch for {suite_path}: "
                f"expected {suite_sha256}, got {actual_suite_sha256}."
            ]
        )
    _verify_input_file(license_ledger, "Frozen license ledger")
    for manifest in manifests:
        _verify_input_file(manifest, "Frozen manifest")
    return FrozenB0RunPlan(
        run_id=run_id,
        plan_path=path.resolve(),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        suite_path=suite_path,
        suite_sha256=suite_sha256,
        license_ledger=license_ledger,
        manifests=manifests,
        model_config=model_config,
        training=training,
        outputs=outputs,
    )


def frozen_b0_run_plan_record(plan: FrozenB0RunPlan) -> dict[str, object]:
    """Return the immutable plan fields in a JSON/checkpoint-safe representation."""

    return {
        "schema_version": FROZEN_B0_RUN_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "purpose": "research",
        "plan_path": str(plan.plan_path),
        "plan_sha256": plan.plan_sha256,
        "suite": {"path": str(plan.suite_path), "sha256": plan.suite_sha256},
        "license_ledger": {
            "path": str(plan.license_ledger.path),
            "sha256": plan.license_ledger.sha256,
        },
        "manifests": [
            {"path": str(manifest.path), "sha256": manifest.sha256} for manifest in plan.manifests
        ],
        "model": {"name": "b0_logmel_cnn", "config": asdict(plan.model_config)},
        "training": asdict(plan.training),
        "outputs": {
            "checkpoint": str(plan.outputs.checkpoint),
            "report": str(plan.outputs.report),
        },
    }


def validate_frozen_b0_suite_inputs(plan: FrozenB0RunPlan, suite: UnseenGeneratorSuite) -> None:
    """Require the plan to pin exactly every manifest referenced by its suite."""

    expected_paths = {
        suite.train.manifest_path,
        suite.dev.manifest_path,
        *(test.selection.manifest_path for test in suite.final_tests),
    }
    pinned_paths = {manifest.path for manifest in plan.manifests}
    missing = sorted(str(path) for path in expected_paths.difference(pinned_paths))
    extra = sorted(str(path) for path in pinned_paths.difference(expected_paths))
    issues: list[str] = []
    if missing:
        issues.append("Frozen run plan does not pin suite manifests: " + ", ".join(missing) + ".")
    if extra:
        issues.append("Frozen run plan pins manifests outside its suite: " + ", ".join(extra) + ".")
    if issues:
        raise FrozenB0RunPlanError(issues)


def epoch_metrics(result: EpochResult) -> dict[str, object]:
    """Serialize all aggregate B0 metrics without dropping exact class counts."""

    from kds.eval.metrics import classification_confidence_intervals

    return {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "correct": result.correct,
        "examples": result.examples,
        "bonafide_examples": result.bonafide_examples,
        "spoof_examples": result.spoof_examples,
        "bonafide_accuracy": result.bonafide_accuracy,
        "bonafide_correct": result.bonafide_correct,
        "spoof_accuracy": result.spoof_accuracy,
        "spoof_correct": result.spoof_correct,
        "balanced_accuracy": result.balanced_accuracy,
        "confidence_intervals": {
            name: asdict(interval)
            for name, interval in classification_confidence_intervals(
                correct=result.correct,
                examples=result.examples,
                bonafide_correct=result.bonafide_correct,
                bonafide_examples=result.bonafide_examples,
                spoof_correct=result.spoof_correct,
                spoof_examples=result.spoof_examples,
            ).items()
        },
    }


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash tensor names, metadata and bytes in a stable order before final evaluation."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        data = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _parse_suite(value: object, base_directory: Path) -> tuple[Path, str]:
    raw = _object(value, "suite")
    _expect_exact_keys(raw, {"path", "sha256"}, "suite")
    suite_path = _relative_path(raw, "path", "suite", base_directory)
    sha256 = _required_string(raw, "sha256", "suite")
    if _SHA256.fullmatch(sha256) is None:
        raise FrozenB0RunPlanError(["suite sha256 must contain 64 lowercase hexadecimal digits."])
    return suite_path, sha256


def _parse_input_file(value: object, label: str, base_directory: Path) -> FrozenInputFile:
    raw = _object(value, label)
    _expect_exact_keys(raw, {"path", "sha256"}, label)
    sha256 = _required_string(raw, "sha256", label)
    if _SHA256.fullmatch(sha256) is None:
        raise FrozenB0RunPlanError(
            [f"{label} sha256 must contain 64 lowercase hexadecimal digits."]
        )
    return FrozenInputFile(
        path=_relative_path(raw, "path", label, base_directory),
        sha256=sha256,
    )


def _parse_manifests(value: object, base_directory: Path) -> tuple[FrozenInputFile, ...]:
    if not isinstance(value, list) or not value:
        raise FrozenB0RunPlanError(["manifests must be a non-empty JSON array."])
    manifests = tuple(
        _parse_input_file(item, f"manifest {index}", base_directory)
        for index, item in enumerate(value, start=1)
    )
    paths = [manifest.path for manifest in manifests]
    if len(paths) != len(set(paths)):
        raise FrozenB0RunPlanError(["manifests must not contain duplicate paths."])
    return manifests


def _parse_model(value: object) -> B0Config:
    raw = _object(value, "model")
    _expect_exact_keys(raw, {"name", "config"}, "model")
    if _required_string(raw, "name", "model") != "b0_logmel_cnn":
        raise FrozenB0RunPlanError(["Frozen B0 runner supports model name 'b0_logmel_cnn' only."])
    config = _object(raw["config"], "model config")
    _expect_exact_keys(
        config, {"sample_rate", "n_fft", "hop_length", "n_mels", "dropout"}, "model config"
    )
    try:
        return B0Config(
            sample_rate=_required_int(config, "sample_rate", "model config", minimum=1),
            n_fft=_required_int(config, "n_fft", "model config", minimum=1),
            hop_length=_required_int(config, "hop_length", "model config", minimum=1),
            n_mels=_required_int(config, "n_mels", "model config", minimum=1),
            dropout=_required_float(config, "dropout", "model config", minimum=0.0),
        )
    except ValueError as error:
        raise FrozenB0RunPlanError([f"Invalid model config: {error}"]) from error


def _parse_training(value: object) -> FrozenB0TrainingConfig:
    raw = _object(value, "training")
    _expect_exact_keys(
        raw,
        {
            "seed",
            "epochs",
            "batch_size",
            "window_samples",
            "learning_rate",
            "weight_decay",
            "num_workers",
            "device",
        },
        "training",
    )
    device = _required_string(raw, "device", "training")
    if device not in {"cpu", "cuda"}:
        raise FrozenB0RunPlanError(["training device must be exactly 'cpu' or 'cuda'."])
    return FrozenB0TrainingConfig(
        seed=_required_int(raw, "seed", "training", minimum=0),
        epochs=_required_int(raw, "epochs", "training", minimum=1),
        batch_size=_required_int(raw, "batch_size", "training", minimum=1),
        window_samples=_required_int(raw, "window_samples", "training", minimum=1),
        learning_rate=_required_float(
            raw, "learning_rate", "training", minimum=0.0, minimum_inclusive=False
        ),
        weight_decay=_required_float(raw, "weight_decay", "training", minimum=0.0),
        num_workers=_required_int(raw, "num_workers", "training", minimum=0),
        device=device,
    )


def _parse_outputs(value: object, base_directory: Path) -> FrozenB0Outputs:
    raw = _object(value, "outputs")
    _expect_exact_keys(raw, {"checkpoint", "report"}, "outputs")
    return FrozenB0Outputs(
        checkpoint=_relative_path(raw, "checkpoint", "outputs", base_directory),
        report=_relative_path(raw, "report", "outputs", base_directory),
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FrozenB0RunPlanError([f"{label} must be a JSON object."])
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
        raise FrozenB0RunPlanError(issues)


def _required_string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise FrozenB0RunPlanError([f"{label} field {name!r} must be a non-empty string."])
    return value.strip()


def _required_int(raw: Mapping[str, object], name: str, label: str, *, minimum: int) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FrozenB0RunPlanError(
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
) -> float:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenB0RunPlanError([f"{label} field {name!r} must be a finite number."])
    result = float(value)
    below_minimum = result < minimum if minimum_inclusive else result <= minimum
    if not math.isfinite(result) or below_minimum:
        operator = "greater than or equal to" if minimum_inclusive else "greater than"
        raise FrozenB0RunPlanError(
            [f"{label} field {name!r} must be finite and {operator} {minimum}."]
        )
    return result


def _relative_path(raw: Mapping[str, object], name: str, label: str, base_directory: Path) -> Path:
    value = Path(_required_string(raw, name, label))
    if value.is_absolute():
        raise FrozenB0RunPlanError([f"{label} field {name!r} must be a relative path."])
    return (base_directory / value).resolve()


def _file_sha256(path: Path, label: str) -> str:
    if not path.is_file():
        raise FrozenB0RunPlanError([f"{label} does not exist: {path}"])
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FrozenB0RunPlanError([f"Cannot hash {label.lower()} {path}: {error}"]) from error
    return digest.hexdigest()


def _verify_input_file(input_file: FrozenInputFile, label: str) -> None:
    actual_sha256 = _file_sha256(input_file.path, label)
    if actual_sha256 != input_file.sha256:
        raise FrozenB0RunPlanError(
            [
                f"{label} SHA-256 mismatch for {input_file.path}: "
                f"expected {input_file.sha256}, got {actual_sha256}."
            ]
        )
