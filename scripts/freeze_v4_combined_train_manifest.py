"""Freeze the v4 20,000-row train manifest without authorizing model training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import AssetValidationError, require_valid_assets
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, write_manifest
from kds.data.v4_train_manifest import (
    V4TrainAssemblyReport,
    V4TrainManifestError,
    build_v4_combined_train_manifest,
)

PROTOCOL_ID = "xlsr-sls-model-v4-combined-train-manifest-v1"
PLAN_SCHEMA_VERSION = 1
_SHA256_HEX = frozenset("0123456789abcdef")


class CombinedTrainManifestPlanError(ValueError):
    """Raised when the frozen combined-train manifest contract cannot be used safely."""


@dataclass(frozen=True, slots=True)
class Binding:
    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class Plan:
    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, Binding]
    expected_source_cells: Mapping[str, int]
    expected_spoof_cells: Mapping[str, int]
    expected_combined_cells: Mapping[str, int]
    expected_shared_text_hashes: int
    output_manifest: str
    output_receipt: str


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CombinedTrainManifestPlanError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise CombinedTrainManifestPlanError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CombinedTrainManifestPlanError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CombinedTrainManifestPlanError(f"{label} must be a non-empty project-relative path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise CombinedTrainManifestPlanError(f"{label} must be a safe project-relative path.")
    return parsed.as_posix()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise CombinedTrainManifestPlanError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CombinedTrainManifestPlanError(f"{label} must be a non-negative integer.")
    return value


def _binding(value: object, label: str) -> Binding:
    raw = _mapping(value, label)
    if set(raw) != {"path", "sha256", "rows"}:
        raise CombinedTrainManifestPlanError(
            f"{label} must contain exactly path, sha256 and rows."
        )
    rows_raw = raw["rows"]
    if rows_raw is not None:
        _nonnegative_int(rows_raw, f"{label}.rows")
    return Binding(
        path=_safe_path(raw["path"], f"{label}.path"),
        sha256=_sha256(raw["sha256"], f"{label}.sha256"),
        rows=cast(int | None, rows_raw),
    )


def _cell_counts(value: object, label: str) -> dict[str, int]:
    raw = _mapping(value, label)
    result: dict[str, int] = {}
    for key, count in raw.items():
        if key not in {"kk/bonafide", "kk/spoof", "ru/bonafide", "ru/spoof"}:
            raise CombinedTrainManifestPlanError(f"{label} has unsupported cell {key!r}.")
        result[key] = _nonnegative_int(count, f"{label}.{key}")
    if not result:
        raise CombinedTrainManifestPlanError(f"{label} must not be empty.")
    return dict(sorted(result.items()))


def _project_path(project_root: Path, relative_path: str, label: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise CombinedTrainManifestPlanError(
            f"{label} resolves outside the project root."
        ) from error
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CombinedTrainManifestPlanError(f"Cannot count CSV rows in {path}.") from error


def _verify_binding(binding: Binding, project_root: Path, label: str) -> Path:
    path = _project_path(project_root, binding.path, label)
    if not path.is_file():
        raise CombinedTrainManifestPlanError(f"{label} is missing: {binding.path}")
    if _file_sha256(path) != binding.sha256:
        raise CombinedTrainManifestPlanError(f"{label} SHA-256 mismatch: {binding.path}")
    if binding.rows is not None and _csv_rows(path) != binding.rows:
        raise CombinedTrainManifestPlanError(f"{label} row count mismatch: {binding.path}")
    return path


def load_plan(path: Path, project_root: Path) -> Plan:
    """Load and hash-verify the pre-output combined-train manifest contract."""

    root = project_root.resolve()
    plan_path = path.resolve()
    try:
        relative_plan = plan_path.relative_to(root).as_posix()
    except ValueError as error:
        raise CombinedTrainManifestPlanError("Plan must be inside the project root.") from error
    raw = _json_object(plan_path, "combined-train manifest plan")
    expected_top = {
        "schema_version",
        "protocol_id",
        "created_at",
        "inputs",
        "expected",
        "restrictions",
        "outputs",
    }
    if set(raw) != expected_top:
        raise CombinedTrainManifestPlanError("Combined-train manifest plan has unexpected keys.")
    if raw["schema_version"] != PLAN_SCHEMA_VERSION:
        raise CombinedTrainManifestPlanError("Unsupported combined-train manifest plan schema.")
    if raw["protocol_id"] != PROTOCOL_ID:
        raise CombinedTrainManifestPlanError("Combined-train manifest protocol_id mismatch.")
    if not isinstance(raw["created_at"], str) or not raw["created_at"]:
        raise CombinedTrainManifestPlanError("created_at must be a non-empty string.")

    expected_inputs = {
        "source_frozen_manifest",
        "source_decode_receipt",
        "kk_spoof_frozen_manifest",
        "kk_spoof_audio_gate_receipt",
        "kk_spoof_audio_gate_governance",
        "license_ledger",
        "manifest_module",
        "runner_script",
    }
    raw_inputs = _mapping(raw["inputs"], "inputs")
    if set(raw_inputs) != expected_inputs:
        raise CombinedTrainManifestPlanError("Combined-train manifest plan input keys are invalid.")
    inputs = {name: _binding(value, f"inputs.{name}") for name, value in raw_inputs.items()}

    expected_raw = _mapping(raw["expected"], "expected")
    if set(expected_raw) != {
        "source_cell_counts",
        "spoof_cell_counts",
        "combined_cell_counts",
        "shared_text_hashes",
    }:
        raise CombinedTrainManifestPlanError("Combined-train expected block has unexpected keys.")
    expected_source_cells = _cell_counts(expected_raw["source_cell_counts"], "source_cell_counts")
    expected_spoof_cells = _cell_counts(expected_raw["spoof_cell_counts"], "spoof_cell_counts")
    expected_combined_cells = _cell_counts(
        expected_raw["combined_cell_counts"], "combined_cell_counts"
    )
    shared_text_hashes = _nonnegative_int(expected_raw["shared_text_hashes"], "shared_text_hashes")

    restrictions = _mapping(raw["restrictions"], "restrictions")
    required_restrictions = {
        "detector_or_logit_feedback": False,
        "new_audio_generation": False,
        "audio_mutation": False,
        "output_overwrite": False,
        "actual_training_execution": False,
        "checkpoint_selection": False,
        "calibration": False,
        "final_inference": False,
    }
    if restrictions != required_restrictions:
        raise CombinedTrainManifestPlanError(
            "Combined-train manifest restrictions are not fail-closed."
        )

    outputs = _mapping(raw["outputs"], "outputs")
    if set(outputs) != {"combined_manifest", "receipt"}:
        raise CombinedTrainManifestPlanError("Combined-train manifest outputs are invalid.")
    output_manifest = _safe_path(outputs["combined_manifest"], "outputs.combined_manifest")
    output_receipt = _safe_path(outputs["receipt"], "outputs.receipt")
    if output_manifest == output_receipt:
        raise CombinedTrainManifestPlanError("Manifest and receipt output paths must differ.")

    for name, binding in inputs.items():
        _verify_binding(binding, root, f"inputs.{name}")
    return Plan(
        path=relative_plan,
        sha256=_file_sha256(plan_path),
        created_at=raw["created_at"],
        inputs=inputs,
        expected_source_cells=expected_source_cells,
        expected_spoof_cells=expected_spoof_cells,
        expected_combined_cells=expected_combined_cells,
        expected_shared_text_hashes=shared_text_hashes,
        output_manifest=output_manifest,
        output_receipt=output_receipt,
    )


def _require_receipt_bindings(plan: Plan, project_root: Path) -> None:
    source_receipt_path = _project_path(
        project_root, plan.inputs["source_decode_receipt"].path, "source decode receipt"
    )
    source_receipt = _json_object(source_receipt_path, "source decode receipt")
    if source_receipt.get("state") != "source_train_frozen_15000_kk_spoof_synthesis_authorized":
        raise CombinedTrainManifestPlanError("Source decode receipt has an invalid state.")
    source_claims = _mapping(source_receipt.get("claims"), "source decode receipt claims")
    if source_claims.get("training_authorized") is not False:
        raise CombinedTrainManifestPlanError("Source decode receipt must not authorize training.")
    source_outputs = _mapping(source_receipt.get("outputs"), "source decode receipt outputs")
    source_frozen = _mapping(
        source_outputs.get("frozen_source_train_manifest"), "source frozen manifest binding"
    )
    if (
        source_frozen.get("path") != plan.inputs["source_frozen_manifest"].path
        or source_frozen.get("sha256") != plan.inputs["source_frozen_manifest"].sha256
        or source_frozen.get("rows") != plan.inputs["source_frozen_manifest"].rows
    ):
        raise CombinedTrainManifestPlanError(
            "Source receipt does not bind the frozen source manifest."
        )

    gate_receipt_path = _project_path(
        project_root, plan.inputs["kk_spoof_audio_gate_receipt"].path, "KK spoof audio gate receipt"
    )
    gate_receipt = _json_object(gate_receipt_path, "KK spoof audio gate receipt")
    gate_outputs = _mapping(gate_receipt.get("outputs"), "KK spoof audio gate outputs")
    spoof_frozen = _mapping(
        gate_outputs.get("frozen_kk_spoof_train_manifest"), "KK spoof frozen manifest binding"
    )
    if (
        spoof_frozen.get("path") != plan.inputs["kk_spoof_frozen_manifest"].path
        or spoof_frozen.get("sha256") != plan.inputs["kk_spoof_frozen_manifest"].sha256
        or spoof_frozen.get("rows") != plan.inputs["kk_spoof_frozen_manifest"].rows
    ):
        raise CombinedTrainManifestPlanError(
            "KK spoof gate does not bind the frozen spoof manifest."
        )

    governance_path = _project_path(
        project_root,
        plan.inputs["kk_spoof_audio_gate_governance"].path,
        "KK spoof audio gate governance",
    )
    governance = _json_object(governance_path, "KK spoof audio gate governance")
    affected = _mapping(governance.get("affected_write_once_receipt"), "affected gate receipt")
    reconciliation = _mapping(governance.get("receipt_reconciliation"), "gate reconciliation")
    authorizations = _mapping(governance.get("authorizations"), "gate authorizations")
    if (
        affected.get("path") != plan.inputs["kk_spoof_audio_gate_receipt"].path
        or affected.get("sha256") != plan.inputs["kk_spoof_audio_gate_receipt"].sha256
        or reconciliation.get("effective_actual_training_authorized") is not False
        or authorizations.get("combined_train_manifest_construction") is not True
        or authorizations.get("actual_training_execution") is not False
    ):
        raise CombinedTrainManifestPlanError("KK spoof gate governance authorization is invalid.")


def validate_inputs(
    plan: Plan, project_root: Path
) -> tuple[tuple[ManifestRow, ...], tuple[ManifestRow, ...], V4TrainAssemblyReport]:
    """Validate pinned receipts, rights and frozen-input invariants without writing outputs."""

    _require_receipt_bindings(plan, project_root)
    source_path = _project_path(
        project_root, plan.inputs["source_frozen_manifest"].path, "source frozen manifest"
    )
    spoof_path = _project_path(
        project_root, plan.inputs["kk_spoof_frozen_manifest"].path, "KK spoof frozen manifest"
    )
    try:
        source_rows = tuple(load_manifest(source_path))
        spoof_rows = tuple(load_manifest(spoof_path))
    except ManifestError as error:
        raise CombinedTrainManifestPlanError(error.issues) from error
    try:
        combined_rows, report = build_v4_combined_train_manifest(
            source_rows,
            spoof_rows,
            expected_source_cells=plan.expected_source_cells,
            expected_spoof_cells=plan.expected_spoof_cells,
            expected_combined_cells=plan.expected_combined_cells,
            expected_shared_text_hashes=plan.expected_shared_text_hashes,
        )
    except V4TrainManifestError as error:
        raise CombinedTrainManifestPlanError(str(error)) from error
    ledger_path = _project_path(project_root, plan.inputs["license_ledger"].path, "license ledger")
    try:
        ledger = load_license_ledger(ledger_path)
        validate_manifest_licenses(combined_rows, ledger)
    except LicenseLedgerError as error:
        raise CombinedTrainManifestPlanError(error.issues) from error
    for source_name in sorted({row.source_name for row in combined_rows}):
        entry = ledger[source_name]
        if entry.train_dev_test_use not in {"research_only", "product_allowed"}:
            raise CombinedTrainManifestPlanError(
                f"Source {source_name!r} is not allowed in research training."
            )
    return combined_rows, spoof_rows, report


def _receipt(
    plan: Plan, report: V4TrainAssemblyReport, output_manifest: Path, project_root: Path
) -> dict[str, object]:
    bindings = {
        name: {
            "path": binding.path,
            "sha256": binding.sha256,
            "rows": binding.rows,
        }
        for name, binding in sorted(plan.inputs.items())
    }
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at": _now(),
        "state": "combined_train_manifest_frozen_training_contract_pending",
        "bindings": {
            "plan": {"path": plan.path, "sha256": plan.sha256},
            **bindings,
        },
        "outputs": {
            "combined_train_manifest": {
                "path": plan.output_manifest,
                "sha256": _file_sha256(output_manifest),
                "rows": report.combined_rows,
            }
        },
        "accounting": asdict(report),
        "claims": {
            "combined_train_manifest_frozen": True,
            "same_train_role_text_overlap_pinned": True,
            "speaker_independence": "not_verified_speaker_independent",
            "training_contract_created": False,
            "actual_training_execution": False,
            "checkpoint_selection": False,
            "calibration": False,
            "final_inference": False,
        },
        "prohibitions": [
            "do not change or rerun either frozen input",
            "do not use detector or logit feedback",
            "do not generate or modify audio",
            "do not train, tune, select a checkpoint, calibrate or run final inference",
        ],
        "next_gate": (
            "create a separate full v4 training contract only after its isolated dev inputs, "
            "runtime, hyperparameters and output paths are hash-pinned; actual training remains "
            "forbidden until that contract passes a no-training preflight"
        ),
    }


def publish(plan: Plan, project_root: Path) -> dict[str, object]:
    """Write the combined manifest and its receipt once, after full asset verification."""

    output_manifest = _project_path(project_root, plan.output_manifest, "combined manifest output")
    output_receipt = _project_path(project_root, plan.output_receipt, "combined receipt output")
    if output_manifest.exists() or output_receipt.exists():
        raise CombinedTrainManifestPlanError(
            "Combined v4 output paths are write-once and already exist."
        )
    combined_rows, _spoof_rows, report = validate_inputs(plan, project_root)
    try:
        require_valid_assets(combined_rows, project_root / "data")
    except AssetValidationError as error:
        raise CombinedTrainManifestPlanError(str(error)) from error
    try:
        write_manifest(output_manifest, combined_rows)
    except ManifestError as error:
        raise CombinedTrainManifestPlanError(error.issues) from error
    receipt = _receipt(plan, report, output_manifest, project_root)
    try:
        with output_receipt.open("x", encoding="utf-8", newline="") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as error:
        raise CombinedTrainManifestPlanError("Cannot publish combined-train receipt.") from error
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/research/v4/xlsr_sls_model_v4_train_manifest_v1.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--publish", action="store_true")
    arguments = parser.parse_args()
    try:
        root = arguments.project_root.resolve()
        plan = load_plan(arguments.plan, root)
        if arguments.validate_only:
            _combined_rows, _spoof_rows, report = validate_inputs(plan, root)
            print(json.dumps({"status": "validated", "accounting": asdict(report)}, sort_keys=True))
        else:
            receipt = publish(plan, root)
            print(json.dumps({"status": "complete", "receipt": receipt["outputs"]}, sort_keys=True))
    except (CombinedTrainManifestPlanError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
