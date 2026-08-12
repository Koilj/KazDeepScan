"""Audit the frozen Stage-C candidate against every manifest referenced by prior run configs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest


class StageCCandidateExposureError(ValueError):
    """Raised when the project-exposure scope or candidate binding is incomplete."""


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageCCandidateExposureError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise StageCCandidateExposureError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _manifest_values(value: object) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "manifest":
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    result.append(cast(str, item["path"]))
            result.extend(_manifest_values(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_manifest_values(item))
    return result


def _resolve_below(project_root: Path, config: Path, value: str) -> Path:
    path = (config.parent / value).resolve(strict=True)
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise StageCCandidateExposureError(
            f"Prior manifest escapes project root: {value!r}."
        ) from error
    if path.suffix != ".csv":
        raise StageCCandidateExposureError(
            f"Prior manifest does not name a CSV: {value!r}."
        )
    return path


def _rows_by_value(rows: list[ManifestRow], field: str) -> dict[str, list[ManifestRow]]:
    result: dict[str, list[ManifestRow]] = {}
    for row in rows:
        result.setdefault(cast(str, getattr(row, field)), []).append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pairing-receipt", type=Path, required=True)
    parser.add_argument("--generator-route-gate", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise StageCCandidateExposureError(
                "Stage-C exposure receipt must be new with an existing parent."
            )
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        project_root = arguments.project_root.resolve(strict=True)
        config_root = arguments.config_root.resolve(strict=True)
        config_root.relative_to(project_root)
        configs = sorted(config_root.glob("*.json"))
        if not configs:
            raise StageCCandidateExposureError("Stage-C exposure audit found no run configs.")
        manifest_paths: set[Path] = set()
        config_bindings: list[dict[str, object]] = []
        for config in configs:
            raw: object = json.loads(config.read_text(encoding="utf-8"))
            values = _manifest_values(raw)
            config_bindings.append(
                {
                    "path": config.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(config),
                    "manifest_references": len(values),
                }
            )
            for value in values:
                manifest_paths.add(_resolve_below(project_root, config, value))
        if not manifest_paths:
            raise StageCCandidateExposureError(
                "Stage-C exposure audit found no prior manifest references."
            )

        candidate = load_manifest(arguments.candidate)
        validate_manifest(candidate)
        if len(candidate) != 334:
            raise StageCCandidateExposureError("Stage-C candidate must contain 334 assets.")
        pairing = _json_object(arguments.pairing_receipt, "Stage-C pairing receipt")
        outputs = pairing.get("outputs")
        combined = outputs.get("combined") if isinstance(outputs, dict) else None
        if (
            pairing.get("protocol_id") != "fresh-suite-stage-c-kazakhtts-pairing-v1"
            or not isinstance(combined, dict)
            or combined.get("path") != arguments.candidate.as_posix()
            or combined.get("sha256") != sha256_file(arguments.candidate)
        ):
            raise StageCCandidateExposureError("Stage-C pairing receipt binding is invalid.")
        route_gate = _json_object(arguments.generator_route_gate, "generator route gate")
        audit = route_gate.get("audit")
        if (
            route_gate.get("protocol_id") != "fresh-suite-stage-c-generator-route-gate-v1"
            or not isinstance(audit, dict)
            or audit.get("exact_route_overlap_rows") != 0
        ):
            raise StageCCandidateExposureError("Stage-C generator route gate is invalid.")
        aliases = audit.get("fixed_voice_alias_overlap")
        male_2 = aliases.get("ISSAI_KazakhTTS2_M2") if isinstance(aliases, dict) else None
        if not isinstance(male_2, dict) or male_2.get("rows") != 312:
            raise StageCCandidateExposureError(
                "Stage-C generator route gate lacks the known Male2 alias overlap."
            )

        prior: list[ManifestRow] = []
        exposure_bindings: list[dict[str, object]] = []
        for path in sorted(manifest_paths):
            rows = load_manifest(path)
            validate_manifest(rows)
            prior.extend(rows)
            exposure_bindings.append(
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "sha256": sha256_file(path),
                    "rows": len(rows),
                }
            )
        candidate_maps = {
            field: _rows_by_value(candidate, field)
            for field in ("sample_id", "sha256", "text_hash")
        }
        prior_maps = {
            field: _rows_by_value(prior, field)
            for field in ("sample_id", "sha256", "text_hash")
        }
        overlaps: dict[str, list[str]] = {}
        for field in candidate_maps:
            overlaps[field] = sorted(set(candidate_maps[field]).intersection(prior_maps[field]))
        if any(overlaps.values()):
            raise StageCCandidateExposureError(
                "Stage-C candidate overlaps a prior configured role: "
                + ", ".join(f"{field}={len(values)}" for field, values in overlaps.items())
            )
        payload = {
            "schema_version": 1,
            "protocol_id": "fresh-suite-stage-c-candidate-project-exposure-v1",
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
            "generator_route_gate": {
                "path": arguments.generator_route_gate.as_posix(),
                "sha256": sha256_file(arguments.generator_route_gate),
                "exact_route_overlap_rows": 0,
                "voice_alias_overlap_rows": 312,
            },
            "scope": {
                "configuration_directory": arguments.config_root.as_posix(),
                "configuration_files": config_bindings,
                "referenced_manifests": exposure_bindings,
                "prior_rows": len(prior),
                "comparison_fields": ["sample_id", "sha256", "text_hash"],
            },
            "overlap_counts": {field: len(values) for field, values in overlaps.items()},
            "overlaps": overlaps,
            "claims": {
                "exact_assets_absent_from_prior_configured_roles": True,
                "exact_texts_absent_from_prior_configured_roles": True,
                "exact_generator_route_absent_from_prior_spoof_manifests": True,
                "source_independent": False,
                "speaker_independent": False,
                "voice_alias_overlap_disclosed": True,
            },
            "detector_inference_performed": False,
            "detector_inference_authorized": False,
        }
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        json.JSONDecodeError,
        ManifestError,
        OSError,
        StageCCandidateExposureError,
        ValueError,
    ) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "configuration_files": len(configs),
                "referenced_manifests": len(manifest_paths),
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
