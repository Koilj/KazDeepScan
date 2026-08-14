"""Materialize exact RuASD/KSC2 source bytes from the canonical v4 v2 selection."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.ksc2 import Ksc2AuditError, extract_ksc2_selected_audio
from kds.data.ruasd_catalog import RuAsdCatalogError, load_ruasd_artifact_catalog
from kds.data.ruasd_research import (
    RuAsdResearchError,
    extract_ruasd_research_slice,
)
from kds.data.v4_materialization import (
    V4MaterializationError,
    V4MaterializationProgressCallback,
    bind_v4_ruasd_records,
    decide_v4_raw_exact_eligibility,
    inspect_v4_raw_asset,
    inventory_exposure_binding,
    load_v4_source_candidates,
    publish_v4_raw_materialization,
    v4_raw_manifest_rows,
)
from kds.data.v4_selection import load_v4_exposure_inventory


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4MaterializationError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4MaterializationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4MaterializationError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4MaterializationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _progress(stage: str) -> V4MaterializationProgressCallback:
    def emit(completed: int, total: int, item: str) -> None:
        print(
            json.dumps(
                {
                    "status": "progress",
                    "stage": stage,
                    "completed": completed,
                    "total": total,
                    "item": item,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    return emit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--governance-receipt", type=Path, required=True)
    parser.add_argument("--ruasd-archive-dir", type=Path, required=True)
    parser.add_argument("--ruasd-catalog", type=Path, required=True)
    parser.add_argument("--ruasd-audit", type=Path, required=True)
    parser.add_argument("--ksc2-parts-directory", type=Path, required=True)
    parser.add_argument("--ksc2-audit", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--raw-destination", type=Path, required=True)
    parser.add_argument("--output-inventory", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        raw_destination = arguments.raw_destination.resolve()
        raw_destination.relative_to(data_root)
        candidates = load_v4_source_candidates(
            arguments.candidate_csv, arguments.governance_receipt
        )
        governance = _json_object(arguments.governance_receipt, "selection governance")
        canonical = _mapping(governance.get("canonical_packet"), "canonical packet")
        selection_binding = _mapping(canonical.get("receipt"), "selection receipt binding")
        if (
            selection_binding.get("path") != arguments.selection_receipt.as_posix()
            or sha256_file(arguments.selection_receipt)
            != _sha256(selection_binding.get("sha256"), "selection receipt")
        ):
            raise V4MaterializationError("Canonical selection receipt binding is invalid.")
        selection_receipt = _json_object(arguments.selection_receipt, "selection receipt")
        bindings = _mapping(selection_receipt.get("bindings"), "selection bindings")
        source_bindings = _mapping(bindings.get("sources"), "source bindings")
        ruasd_binding = _mapping(source_bindings.get("ruasd_audit"), "RuASD binding")
        ksc2_binding = _mapping(source_bindings.get("ksc2_audit"), "KSC2 binding")
        config_binding = _mapping(bindings.get("config"), "config binding")
        if (
            ruasd_binding.get("path") != arguments.ruasd_audit.as_posix()
            or sha256_file(arguments.ruasd_audit)
            != _sha256(ruasd_binding.get("sha256"), "RuASD audit")
            or ksc2_binding.get("path") != arguments.ksc2_audit.as_posix()
            or sha256_file(arguments.ksc2_audit)
            != _sha256(ksc2_binding.get("sha256"), "KSC2 audit")
        ):
            raise V4MaterializationError("Selection source-audit bindings changed.")
        ruasd_audit = _json_object(arguments.ruasd_audit, "RuASD audit")
        ksc2_audit = _json_object(arguments.ksc2_audit, "KSC2 audit")
        compressed_hash = _sha256(
            ksc2_audit.get("compressed_sha256"), "KSC2 compressed archive"
        )
        if (
            ruasd_audit.get("archive_count") != 250
            or ruasd_audit.get("sha256_verified_archives") != 250
            or ksc2_binding.get("compressed_sha256") != compressed_hash
        ):
            raise V4MaterializationError("Source audits are incomplete or inconsistent.")
        exposure = load_v4_exposure_inventory(arguments.manifest_root, project_root)
        ruasd_candidates = tuple(
            candidate for candidate in candidates if candidate.source_id == "ruasd_ru_v1_full"
        )
        ksc2_candidates = tuple(
            candidate for candidate in candidates if candidate.source_id == "ksc2_v1"
        )
        catalog = load_ruasd_artifact_catalog(arguments.ruasd_catalog)
        records = bind_v4_ruasd_records(
            ruasd_candidates,
            arguments.ruasd_archive_dir,
            catalog,
            progress_callback=_progress("ruasd_metadata_binding"),
        )
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-source-raw-", dir=raw_destination.parent
        ) as stage_name:
            stage = Path(stage_name)
            payload = stage / "payload"
            payload.mkdir()
            ruasd_output = payload / "ruasd"
            ruasd_extracted = extract_ruasd_research_slice(
                arguments.ruasd_archive_dir,
                catalog,
                records,
                ruasd_output,
                progress_callback=_progress("ruasd_audio_extraction"),
            )
            ksc2_output = payload / "ksc2"
            ksc2_extracted = extract_ksc2_selected_audio(
                arguments.ksc2_parts_directory,
                ksc2_output,
                selected_members=frozenset(
                    candidate.archive_audio_member for candidate in ksc2_candidates
                ),
                expected_compressed_sha256=compressed_hash,
                progress_callback=_progress("ksc2_audio_extraction"),
            )
            ksc2_by_member = {item.archive_member: item for item in ksc2_extracted}
            record_by_candidate = {
                f"ruasd_ru_v1_full:{record.record_key}": record for record in records
            }
            assets = []
            raw_prefix = raw_destination.relative_to(data_root)
            for completed, candidate in enumerate(candidates, start=1):
                if candidate.source_id == "ruasd_ru_v1_full":
                    record = record_by_candidate[candidate.candidate_id]
                    path = ruasd_extracted[record.record_key]
                else:
                    extracted = ksc2_by_member[candidate.archive_audio_member]
                    path = ksc2_output / extracted.relative_path
                internal = path.relative_to(payload)
                assets.append(
                    inspect_v4_raw_asset(
                        candidate,
                        path,
                        (raw_prefix / internal).as_posix(),
                    )
                )
                if completed % 500 == 0 or completed == len(candidates):
                    _progress("raw_audio_inspection")(
                        completed, len(candidates), candidate.candidate_id
                    )
            decisions = decide_v4_raw_exact_eligibility(
                assets, exposure.audio_sha256
            )
            raw_rows = v4_raw_manifest_rows(decisions, created_at=arguments.created_at)
            receipt_bindings: dict[str, object] = {
                "candidate_csv": {
                    "path": arguments.candidate_csv.as_posix(),
                    "sha256": sha256_file(arguments.candidate_csv),
                },
                "selection_receipt": {
                    "path": arguments.selection_receipt.as_posix(),
                    "sha256": sha256_file(arguments.selection_receipt),
                },
                "selection_governance": {
                    "path": arguments.governance_receipt.as_posix(),
                    "sha256": sha256_file(arguments.governance_receipt),
                },
                "config": config_binding,
                "ruasd_catalog": {
                    "path": arguments.ruasd_catalog.as_posix(),
                    "sha256": sha256_file(arguments.ruasd_catalog),
                },
                "ruasd_audit": ruasd_binding,
                "ksc2_audit": ksc2_binding,
                "license_ledger": {
                    "path": arguments.license_ledger.as_posix(),
                    "sha256": sha256_file(arguments.license_ledger),
                },
                "historical_exposure": inventory_exposure_binding(exposure),
            }
            publish_v4_raw_materialization(
                raw_destination=raw_destination,
                staged_raw_root=payload,
                inventory_path=arguments.output_inventory,
                raw_manifest_path=arguments.output_manifest,
                receipt_path=arguments.output_receipt,
                decisions=decisions,
                raw_rows=raw_rows,
                data_root=data_root,
                license_ledger_path=arguments.license_ledger,
                created_at=arguments.created_at,
                bindings=receipt_bindings,
            )
    except (
        V4MaterializationError,
        RuAsdCatalogError,
        RuAsdResearchError,
        Ksc2AuditError,
        OSError,
        ValueError,
    ) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    receipt = _json_object(arguments.output_receipt, "published materialization receipt")
    print(
        json.dumps(
            {
                "status": "ok",
                "raw_inventory": arguments.output_inventory.as_posix(),
                "raw_inventory_sha256": sha256_file(arguments.output_inventory),
                "raw_manifest": arguments.output_manifest.as_posix(),
                "raw_manifest_sha256": sha256_file(arguments.output_manifest),
                "receipt": arguments.output_receipt.as_posix(),
                "receipt_sha256": sha256_file(arguments.output_receipt),
                "accounting": receipt["accounting"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
