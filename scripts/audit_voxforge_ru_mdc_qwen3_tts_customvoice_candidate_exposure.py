#!/usr/bin/env python3
"""Audit the completed VoxForge/Qwen candidate against prior configured roles."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.qwen3_tts_customvoice_candidate import QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
from kds.data.voxforge import VOXFORGE_RU_SOURCE_ID

CANDIDATE_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-pairing-v1"
ROUTE_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-exact-route-audit-v1"


class VoxForgeQwenExposureAuditError(ValueError):
    """Raised when the candidate binding or configured-role audit is incomplete."""


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgeQwenExposureAuditError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise VoxForgeQwenExposureAuditError(f"{label} must be a JSON object.")
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
    try:
        path = (config.parent / value).resolve(strict=True)
    except OSError as error:
        raise VoxForgeQwenExposureAuditError(
            f"Cannot resolve prior manifest {value!r} from {config}."
        ) from error
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise VoxForgeQwenExposureAuditError(
            f"Prior manifest escapes project root: {value!r}."
        ) from error
    if path.suffix != ".csv":
        raise VoxForgeQwenExposureAuditError(f"Prior manifest is not a CSV: {value!r}.")
    return path


def _require_candidate(candidate: list[ManifestRow]) -> None:
    if len(candidate) != 158 or Counter(row.label for row in candidate) != Counter(
        {"bonafide": 79, "spoof": 79}
    ):
        raise VoxForgeQwenExposureAuditError(
            "VoxForge/Qwen candidate must contain exactly 79 binary pairs / 158 assets."
        )
    by_text: dict[str, list[ManifestRow]] = {}
    for row in candidate:
        by_text.setdefault(row.text_id, []).append(row)
    if len(by_text) != 79 or any(
        len(pair) != 2
        or {row.label for row in pair} != {"bonafide", "spoof"}
        or len({row.text_hash for row in pair}) != 1
        for pair in by_text.values()
    ):
        raise VoxForgeQwenExposureAuditError(
            "VoxForge/Qwen candidate has invalid exact text pairs."
        )
    if any(
        row.split != "test"
        or row.language != "ru"
        or (
            row.source_name
            != (
                VOXFORGE_RU_SOURCE_ID
                if row.label == "bonafide"
                else QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
            )
        )
        for row in candidate
    ):
        raise VoxForgeQwenExposureAuditError(
            "VoxForge/Qwen candidate must contain only the fixed RU test source pairing."
        )


def _require_pairing_receipt(path: Path, candidate: Path) -> None:
    receipt = _object(path, "VoxForge/Qwen pairing receipt")
    output = receipt.get("output_candidate")
    counts = receipt.get("counts")
    decision_rule = receipt.get("decision_rule")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != CANDIDATE_PROTOCOL_ID
        or not isinstance(output, dict)
        or output.get("path") != candidate.as_posix()
        or output.get("sha256") != sha256_file(candidate)
        or output.get("rows") != 158
        or not isinstance(counts, dict)
        or counts.get("retained_pairs") != 79
        or counts.get("technical_qa_rejected_spoof_rows") != 0
        or not isinstance(decision_rule, dict)
        or decision_rule.get("exact_text_hash_and_text_id_match") is not True
        or decision_rule.get("metric_or_detector_based_selection") is not False
        or decision_rule.get("post_selection_backfill") is not False
        or decision_rule.get("resynthesis_after_qa") is not False
        or decision_rule.get("detector_inference_performed") is not False
        or decision_rule.get("detector_inference_authorized") is not False
    ):
        raise VoxForgeQwenExposureAuditError(
            "VoxForge/Qwen pairing receipt is invalid or does not bind the candidate."
        )


def _require_route_audit(path: Path) -> None:
    route = _object(path, "VoxForge/Qwen exact-route audit")
    claims = route.get("claims")
    gate = route.get("route_gate")
    aliases = gate.get("fixed_voice_alias_overlap") if isinstance(gate, dict) else None
    aiden = aliases.get("aiden") if isinstance(aliases, dict) else None
    qualified = aliases.get("qwen3_tts_customvoice:aiden") if isinstance(aliases, dict) else None
    historical_qwen = route.get("historical_qwen3_identifier_evidence")
    if (
        route.get("schema_version") != 1
        or route.get("protocol_id") != ROUTE_PROTOCOL_ID
        or not isinstance(claims, dict)
        or not isinstance(gate, dict)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("speaker_independence") is not False
        or claims.get("reference_audio_or_voice_cloning_used") is not False
        or gate.get("exact_route_overlap_rows") != 0
        or gate.get("generator_family_overlap_rows") != 0
        or not isinstance(aiden, dict)
        or aiden.get("rows") != 0
        or not isinstance(qualified, dict)
        or qualified.get("rows") != 0
        or not isinstance(historical_qwen, dict)
        or historical_qwen.get("rows") != 0
    ):
        raise VoxForgeQwenExposureAuditError(
            "VoxForge/Qwen exact-route audit is invalid or its limits changed."
        )


def _require_acoustic_gate(path: Path, candidate: list[ManifestRow]) -> None:
    report = _object(path, "VoxForge/Qwen acoustic gate report")
    results = report.get("asset_results")
    expected = {(row.sample_id, row.sha256) for row in candidate}
    passed = (
        {
            (item.get("sample_id"), item.get("audio_sha256"))
            for item in results
            if isinstance(item, dict)
            and item.get("decision") == "pass"
            and item.get("review_count") == 2
        }
        if isinstance(results, list)
        else set()
    )
    if (
        report.get("schema_version") != 1
        or report.get("review_rows") != 316
        or report.get("decision_counts") != {"pass": 158}
        or report.get("all_assets_acoustically_verified") is not True
        or report.get("evaluation_contract_authorized") is not True
        or report.get("detector_inference_performed") is not False
        or passed != expected
    ):
        raise VoxForgeQwenExposureAuditError(
            "VoxForge/Qwen acoustic gate is not a full exact-byte pass for this candidate."
        )


def _values(rows: list[ManifestRow], field: str) -> set[str]:
    return {cast(str, getattr(row, field)) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pairing-receipt", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--acoustic-gate", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise VoxForgeQwenExposureAuditError(
                "Exposure audit output must be new with an existing parent."
            )
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        project_root = arguments.project_root.resolve(strict=True)
        config_root = arguments.config_root.resolve(strict=True)
        config_root.relative_to(project_root)
        configs = sorted(config_root.glob("*.json"))
        if not configs:
            raise VoxForgeQwenExposureAuditError(
                "Exposure audit found no research configs."
            )
        candidate = load_manifest(arguments.candidate)
        validate_manifest(candidate)
        _require_candidate(candidate)
        _require_pairing_receipt(arguments.pairing_receipt, arguments.candidate)
        _require_route_audit(arguments.route_audit)
        _require_acoustic_gate(arguments.acoustic_gate, candidate)

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
            raise VoxForgeQwenExposureAuditError(
                "Exposure audit found no referenced manifests."
            )
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
            raise VoxForgeQwenExposureAuditError(
                "VoxForge/Qwen candidate overlaps a prior configured role: "
                + ", ".join(f"{field}={len(values)}" for field, values in overlaps.items())
            )
        payload = {
            "schema_version": 1,
            "protocol_id": (
                "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-candidate-"
                "project-exposure-v1"
            ),
            "created_at": arguments.created_at,
            "candidate": {
                "path": arguments.candidate.as_posix(),
                "sha256": sha256_file(arguments.candidate),
                "rows": len(candidate),
                "pairs": 79,
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
            "acoustic_gate": {
                "path": arguments.acoustic_gate.as_posix(),
                "sha256": sha256_file(arguments.acoustic_gate),
                "assets_passed": 158,
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
                "vendor_family_independent": False,
                "source_independent": False,
                "speaker_independent": False,
                "russian_native_voice": False,
                "organizational_reviewer_independence_proven": False,
            },
            "detector_inference_performed": False,
            "detector_inference_authorized": False,
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (ManifestError, OSError, VoxForgeQwenExposureAuditError, ValueError) as error:
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
