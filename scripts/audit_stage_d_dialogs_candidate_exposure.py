"""Audit the Stage-D Dialogs-RU candidate against every prior configured manifest role."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest


class StageDExposureAuditError(ValueError):
    """Raised when the prior-role scope or candidate binding is incomplete."""


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageDExposureAuditError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise StageDExposureAuditError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _manifest_values(value: object) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "manifest":
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    values.append(cast(str, item["path"]))
            values.extend(_manifest_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_manifest_values(item))
    return values


def _resolve_manifest(project_root: Path, config: Path, value: str) -> Path:
    path = (config.parent / value).resolve(strict=True)
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise StageDExposureAuditError(
            f"Prior manifest escapes project root: {value!r}."
        ) from error
    if path.suffix != ".csv":
        raise StageDExposureAuditError(f"Prior manifest is not a CSV: {value!r}.")
    return path


def _values(rows: list[ManifestRow], field: str) -> set[str]:
    return {cast(str, getattr(row, field)) for row in rows}


def _require_candidate(candidate: list[ManifestRow]) -> None:
    if len(candidate) != 110:
        raise StageDExposureAuditError("Stage-D candidate must contain exactly 110 assets.")
    labels = {row.label for row in candidate}
    if labels != {"bonafide", "spoof"} or sum(row.label == "spoof" for row in candidate) != 55:
        raise StageDExposureAuditError("Stage-D candidate must contain exactly 55 binary pairs.")
    by_text: dict[str, list[ManifestRow]] = {}
    for row in candidate:
        by_text.setdefault(row.text_id, []).append(row)
    if len(by_text) != 55 or any(
        len(pair) != 2
        or {row.label for row in pair} != {"bonafide", "spoof"}
        or len({row.text_hash for row in pair}) != 1
        for pair in by_text.values()
    ):
        raise StageDExposureAuditError("Stage-D candidate has invalid text pairs.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pairing-receipt", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise StageDExposureAuditError("Exposure audit output must be new and writable.")
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        project_root = arguments.project_root.resolve(strict=True)
        config_root = arguments.config_root.resolve(strict=True)
        config_root.relative_to(project_root)
        configs = sorted(config_root.glob("*.json"))
        if not configs:
            raise StageDExposureAuditError("Exposure audit found no research configs.")
        candidate = load_manifest(arguments.candidate)
        validate_manifest(candidate)
        _require_candidate(candidate)
        pairing = _object(arguments.pairing_receipt, "Stage-D pairing receipt")
        paired_output = pairing.get("output_candidate")
        if (
            pairing.get("protocol_id") != "stage-d-dialogs-ru-masha-neutral-pairing-v1"
            or not isinstance(paired_output, dict)
            or paired_output.get("path") != arguments.candidate.as_posix()
            or paired_output.get("sha256") != sha256_file(arguments.candidate)
            or paired_output.get("rows") != 110
        ):
            raise StageDExposureAuditError("Stage-D pairing receipt does not bind the candidate.")
        route = _object(arguments.route_audit, "Stage-D exact-route audit")
        claims = route.get("claims")
        gate = route.get("route_gate")
        if (
            route.get("protocol_id") != "stage-d-dialogs-ru-vits2-exact-route-audit-v1"
            or not isinstance(claims, dict)
            or not isinstance(gate, dict)
            or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
            or claims.get("architecture_family_novelty") is not False
            or claims.get("speaker_independence") is not False
            or gate.get("exact_route_overlap_rows") != 0
        ):
            raise StageDExposureAuditError("Stage-D exact-route audit is invalid.")
        manifests: set[Path] = set()
        config_bindings: list[dict[str, object]] = []
        for config in configs:
            raw = _object(config, f"research config {config.name}")
            references = _manifest_values(raw)
            config_bindings.append(
                {
                    "path": config.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(config),
                    "manifest_references": len(references),
                }
            )
            manifests.update(
                _resolve_manifest(project_root, config, value) for value in references
            )
        if not manifests:
            raise StageDExposureAuditError("Exposure audit found no referenced manifests.")
        prior: list[ManifestRow] = []
        manifest_bindings: list[dict[str, object]] = []
        for manifest in sorted(manifests):
            rows = load_manifest(manifest)
            validate_manifest(rows)
            prior.extend(rows)
            manifest_bindings.append(
                {
                    "path": manifest.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(manifest),
                    "rows": len(rows),
                }
            )
        overlaps = {
            field: sorted(_values(candidate, field).intersection(_values(prior, field)))
            for field in ("sample_id", "sha256", "text_hash")
        }
        if any(overlaps.values()):
            raise StageDExposureAuditError(
                "Stage-D candidate overlaps a prior configured role: "
                + ", ".join(f"{field}={len(values)}" for field, values in overlaps.items())
            )
        payload = {
            "schema_version": 1,
            "protocol_id": "stage-d-dialogs-ru-masha-neutral-candidate-project-exposure-v1",
            "created_at": arguments.created_at,
            "candidate": {
                "path": arguments.candidate.as_posix(),
                "sha256": sha256_file(arguments.candidate),
                "rows": len(candidate),
            },
            "pairing_receipt": {
                "path": arguments.pairing_receipt.as_posix(),
                "sha256": sha256_file(arguments.pairing_receipt),
            },
            "route_audit": {
                "path": arguments.route_audit.as_posix(),
                "sha256": sha256_file(arguments.route_audit),
                "exact_route_overlap_rows": 0,
            },
            "scope": {
                "configuration_directory": arguments.config_root.as_posix(),
                "configuration_files": config_bindings,
                "referenced_manifests": manifest_bindings,
                "prior_rows": len(prior),
                "comparison_fields": ["sample_id", "sha256", "text_hash"],
            },
            "overlap_counts": {field: len(values) for field, values in overlaps.items()},
            "overlaps": overlaps,
            "claims": {
                "exact_assets_absent_from_prior_configured_roles": True,
                "exact_texts_absent_from_prior_configured_roles": True,
                "exact_generator_route_absent_from_prior_spoof_manifests": True,
                "architecture_family_novelty": False,
                "source_independent": False,
                "speaker_independent": False,
            },
            "detector_inference_performed": False,
            "detector_inference_authorized": False,
        }
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ManifestError, OSError, StageDExposureAuditError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "configuration_files": len(configs),
                "referenced_manifests": len(manifests),
                "prior_rows": len(prior),
                "overlap_counts": payload["overlap_counts"],
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
