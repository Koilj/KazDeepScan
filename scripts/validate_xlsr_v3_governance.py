"""Validate and publish the write-once XLS-R v3 data-governance receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.training.xlsr_stage_a_plan import (
    PinnedFile,
    XlsrStageAPlanError,
    _parse_symmetric_augmentation,
    load_xlsr_stage_a_plan,
    validate_and_select_xlsr_stage_a,
)

_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_REQUIRED_ROLES = (
    "train",
    "stage_a_dev",
    "stage_b_dev",
    "calibration",
    "final_stage_d",
)
_ROLE_USES = {
    "train": "training_only",
    "stage_a_dev": "stage_a_model_selection_only",
    "stage_b_dev": "stage_b_model_selection_only",
    "calibration": "temperature_scaling_only",
    "final_stage_d": "final_evaluation_only",
}


class GovernanceError(ValueError):
    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True, slots=True)
class Role:
    name: str
    use: str
    manifest: PinnedFile
    source_split: str
    expected_rows: int
    expected_label_counts: dict[str, int]
    expected_source_ids: tuple[str, ...]
    expected_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Contract:
    path: Path
    sha256: str
    contract_id: str
    license_ledger: PinnedFile
    stage_a_plan: PinnedFile
    roles: tuple[Role, ...]
    augmentation: object
    acoustic_gate: PinnedFile
    pairing_receipt: PinnedFile
    controls: dict[str, str]
    implementation: tuple[PinnedFile, ...]
    output: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(pinned: PinnedFile, label: str) -> None:
    if not pinned.path.is_file():
        raise GovernanceError([f"{label} does not exist: {pinned.path}"])
    actual = _sha256(pinned.path)
    if actual != pinned.sha256:
        raise GovernanceError(
            [f"{label} SHA-256 mismatch: expected {pinned.sha256}, got {actual}."]
        )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GovernanceError([f"{label} must be a JSON object."])
    return cast(dict[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(raw))
    unknown = sorted(set(raw).difference(expected))
    issues: list[str] = []
    if missing:
        issues.append(f"{label} missing fields: {', '.join(missing)}.")
    if unknown:
        issues.append(f"{label} unknown fields: {', '.join(unknown)}.")
    if issues:
        raise GovernanceError(issues)


def _string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise GovernanceError([f"{label} {name!r} must be a non-empty string."])
    return value.strip()


def _int(raw: Mapping[str, object], name: str, label: str, *, minimum: int) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GovernanceError([f"{label} {name!r} must be an integer >= {minimum}."])
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise GovernanceError([f"{label} must be a non-empty string list."])
    result = tuple(sorted(item.strip() for item in cast(list[str], value)))
    if not all(result) or len(result) != len(set(result)):
        raise GovernanceError([f"{label} must be unique non-empty strings."])
    return result


def _pinned(value: object, label: str, base: Path) -> PinnedFile:
    raw = _object(value, label)
    _exact_keys(raw, {"path", "sha256"}, label)
    path = Path(_string(raw, "path", label))
    digest = _string(raw, "sha256", label)
    invalid_digest = len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    )
    if path.is_absolute() or invalid_digest:
        raise GovernanceError([f"{label} must contain a relative path and lowercase SHA-256."])
    return PinnedFile(path=(base / path).resolve(), sha256=digest)


def _role(value: object, base: Path) -> Role:
    raw = _object(value, "v3 role")
    _exact_keys(
        raw,
        {
            "name",
            "use",
            "manifest",
            "source_split",
            "expected_rows",
            "expected_label_counts",
            "expected_source_ids",
            "expected_languages",
        },
        "v3 role",
    )
    name = _string(raw, "name", "v3 role")
    labels = _object(raw["expected_label_counts"], f"role {name} expected_label_counts")
    _exact_keys(labels, {"bonafide", "spoof"}, f"role {name} expected_label_counts")
    counts = {
        label: _int(labels, label, f"role {name} expected_label_counts", minimum=1)
        for label in ("bonafide", "spoof")
    }
    expected_rows = _int(raw, "expected_rows", f"role {name}", minimum=2)
    if sum(counts.values()) != expected_rows:
        raise GovernanceError([f"role {name} expected label counts do not sum to expected_rows."])
    return Role(
        name=name,
        use=_string(raw, "use", f"role {name}"),
        manifest=_pinned(raw["manifest"], f"role {name} manifest", base),
        source_split=_string(raw, "source_split", f"role {name}"),
        expected_rows=expected_rows,
        expected_label_counts=counts,
        expected_source_ids=_strings(raw["expected_source_ids"], f"role {name} sources"),
        expected_languages=_strings(raw["expected_languages"], f"role {name} languages"),
    )


def load_contract(path: Path) -> Contract:
    try:
        content = path.read_bytes()
        raw_value: object = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GovernanceError([f"Cannot read v3 governance contract {path}: {error}"]) from error
    raw = _object(raw_value, "v3 governance contract")
    _exact_keys(
        raw,
        {
            "schema_version",
            "contract_id",
            "purpose",
            "license_ledger",
            "stage_a_plan",
            "roles",
            "augmentation",
            "final_stage_d",
            "controls",
            "implementation",
            "output",
        },
        "v3 governance contract",
    )
    if _int(raw, "schema_version", "v3 governance contract", minimum=1) != _SCHEMA_VERSION:
        raise GovernanceError([f"v3 governance schema_version must be {_SCHEMA_VERSION}."])
    if _string(raw, "purpose", "v3 governance contract") != "research":
        raise GovernanceError(["v3 governance purpose must be 'research'."])
    base = path.resolve().parent
    role_values = raw["roles"]
    if not isinstance(role_values, list) or len(role_values) != len(_REQUIRED_ROLES):
        raise GovernanceError([f"roles must contain exactly {len(_REQUIRED_ROLES)} entries."])
    roles = tuple(_role(value, base) for value in role_values)
    if tuple(role.name for role in roles) != _REQUIRED_ROLES:
        raise GovernanceError([f"roles must be ordered exactly as {_REQUIRED_ROLES!r}."])
    for role in roles:
        if role.use != _ROLE_USES[role.name]:
            raise GovernanceError([f"role {role.name} has an invalid restricted use."])
    final_stage_d = _object(raw["final_stage_d"], "final_stage_d")
    _exact_keys(final_stage_d, {"acoustic_gate", "pairing_receipt"}, "final_stage_d")
    controls = _object(raw["controls"], "controls")
    _exact_keys(
        controls,
        {
            "checkpoint_selection",
            "calibration",
            "stage_d_v2_logits_and_errors",
            "stage_d_final_mutation",
            "v3_final_inference",
        },
        "controls",
    )
    expected_controls = {
        "checkpoint_selection": "stage_a_and_stage_b_dev_loss_only",
        "calibration": "temperature_only_on_pinned_calibration_role",
        "stage_d_v2_logits_and_errors": "prohibited_for_all_v3_decisions",
        "stage_d_final_mutation": "prohibited_no_reselection_no_backfill",
        "v3_final_inference": "new_immutable_plan_one_run_after_v3_dev_selection",
    }
    if {name: _string(controls, name, "controls") for name in controls} != expected_controls:
        raise GovernanceError(["controls do not match the fixed v3 governance policy."])
    implementations_value = raw["implementation"]
    if not isinstance(implementations_value, list) or not implementations_value:
        raise GovernanceError(["implementation must be a non-empty pinned-file list."])
    output = Path(_string(raw, "output", "v3 governance contract"))
    if output.is_absolute():
        raise GovernanceError(["output must be a relative path."])
    return Contract(
        path=path.resolve(),
        sha256=hashlib.sha256(content).hexdigest(),
        contract_id=_string(raw, "contract_id", "v3 governance contract"),
        license_ledger=_pinned(raw["license_ledger"], "license_ledger", base),
        stage_a_plan=_pinned(raw["stage_a_plan"], "stage_a_plan", base),
        roles=roles,
        augmentation=raw["augmentation"],
        acoustic_gate=_pinned(final_stage_d["acoustic_gate"], "final_stage_d acoustic_gate", base),
        pairing_receipt=_pinned(
            final_stage_d["pairing_receipt"], "final_stage_d pairing_receipt", base
        ),
        controls=expected_controls,
        implementation=tuple(
            _pinned(value, "implementation", base) for value in implementations_value
        ),
        output=(base / output).resolve(),
    )


def _role_rows(role: Role, ledger: Mapping[str, object]) -> tuple[ManifestRow, ...]:
    _verify(role.manifest, f"role {role.name} manifest")
    try:
        rows = load_manifest(role.manifest.path)
        selected = tuple(row for row in rows if row.split == role.source_split)
        validate_manifest(selected)
        validate_manifest_licenses(selected, ledger)  # type: ignore[arg-type]
    except (ManifestError, LicenseLedgerError) as error:
        raise GovernanceError([str(error)]) from error
    if len(selected) != role.expected_rows:
        raise GovernanceError(
            [f"role {role.name} has {len(selected)} rows, expected {role.expected_rows}."]
        )
    counts = {label: sum(row.label == label for row in selected) for label in ("bonafide", "spoof")}
    if counts != role.expected_label_counts:
        raise GovernanceError([f"role {role.name} label counts do not match the contract."])
    sources = tuple(sorted({row.source_name for row in selected}))
    languages = tuple(sorted({row.language for row in selected}))
    if sources != role.expected_source_ids or languages != role.expected_languages:
        raise GovernanceError(
            [f"role {role.name} source or language set does not match the contract."]
        )
    return selected


def _overlaps(left: tuple[ManifestRow, ...], right: tuple[ManifestRow, ...]) -> dict[str, int]:
    return {
        field: len(
            {getattr(row, field) for row in left}.intersection(
                getattr(row, field) for row in right
            )
        )
        for field in ("sample_id", "sha256", "text_hash", "parent_group_id")
    }


def _check_stage_a(
    contract: Contract, roles: Mapping[str, tuple[ManifestRow, ...]]
) -> dict[str, object]:
    _verify(contract.stage_a_plan, "stage_a_plan")
    try:
        stage_a = load_xlsr_stage_a_plan(contract.stage_a_plan.path)
        report, selected = validate_and_select_xlsr_stage_a(
            stage_a, load_license_ledger(stage_a.license_ledger.path)
        )
        augmentation = _parse_symmetric_augmentation(contract.augmentation)
    except XlsrStageAPlanError as error:
        raise GovernanceError(error.issues) from error
    if stage_a.training.augmentation != augmentation:
        raise GovernanceError(["Stage-A plan augmentation does not equal the governance policy."])
    if selected.train != roles["train"] or selected.dev != roles["stage_a_dev"]:
        raise GovernanceError(["Stage-A selected rows do not equal the pinned governance roles."])
    return {
        "schema_version": stage_a.schema_version,
        "plan_sha256": stage_a.plan_sha256,
        "augmentation": asdict(augmentation),
        "role_rows": {role.role: role.rows for role in report.roles},
    }


def _check_final_stage_d(contract: Contract) -> dict[str, object]:
    _verify(contract.acoustic_gate, "Stage-D acoustic gate")
    _verify(contract.pairing_receipt, "Stage-D pairing receipt")
    try:
        gate = json.loads(contract.acoustic_gate.path.read_bytes())
        pairing = json.loads(contract.pairing_receipt.path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GovernanceError([f"Cannot read pinned Stage-D receipt: {error}"]) from error
    if not isinstance(gate, dict) or not gate.get("all_assets_acoustically_verified"):
        raise GovernanceError(["Stage-D acoustic gate does not authorize all exact final assets."])
    if not isinstance(pairing, dict) or pairing.get("counts", {}).get("retained_pairs") != 55:
        raise GovernanceError(
            ["Stage-D pairing receipt does not record exactly 55 retained pairs."]
        )
    return {
        "acoustic_gate_sha256": contract.acoustic_gate.sha256,
        "pairing_receipt_sha256": contract.pairing_receipt.sha256,
        "retained_pairs": 55,
    }


def validate_contract(contract: Contract, audio_root: Path) -> dict[str, object]:
    if os.path.lexists(contract.output):
        raise GovernanceError(f"Refusing to overwrite v3 governance receipt: {contract.output}")
    _verify(contract.license_ledger, "license_ledger")
    for implementation in contract.implementation:
        _verify(implementation, "implementation")
    ledger = load_license_ledger(contract.license_ledger.path)
    role_rows = {role.name: _role_rows(role, ledger) for role in contract.roles}
    require_valid_assets([row for rows in role_rows.values() for row in rows], audio_root)
    overlaps: dict[str, dict[str, int]] = {}
    issues: list[str] = []
    for left, right in combinations(_REQUIRED_ROLES, 2):
        counts = _overlaps(role_rows[left], role_rows[right])
        overlaps[f"{left}__{right}"] = counts
        if any(counts.values()):
            issues.append(
                f"v3 governance overlap {left}/{right}: "
                + ", ".join(f"{field}={count}" for field, count in counts.items() if count)
            )
    if issues:
        raise GovernanceError(issues)
    return {
        "schema_version": _SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.sha256,
        "status": "validated",
        "v2_stage_d_logits_or_errors_loaded": False,
        "roles": {
            name: {
                "rows": len(rows),
                "label_counts": {
                    label: sum(row.label == label for row in rows)
                    for label in ("bonafide", "spoof")
                },
            }
            for name, rows in role_rows.items()
        },
        "pairwise_overlap_counts": overlaps,
        "assets_validated": sum(len(rows) for rows in role_rows.values()),
        "stage_a": _check_stage_a(contract, role_rows),
        "final_stage_d": _check_final_stage_d(contract),
        "controls": contract.controls,
    }


def _write_new(path: Path, report: Mapping[str, object]) -> None:
    if not path.parent.is_dir():
        raise GovernanceError([f"Receipt directory does not exist: {path.parent}"])
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
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    arguments = parser.parse_args()
    contract = load_contract(arguments.contract)
    report = validate_contract(contract, arguments.audio_root)
    _write_new(contract.output, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
