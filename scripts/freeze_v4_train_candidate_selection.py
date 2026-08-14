"""Freeze metadata-only v4 train candidates before extraction, synthesis, QA, or training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.ksc2 import Ksc2AuditError, scan_ksc2_text_candidates
from kds.data.ruasd_catalog import RuAsdCatalogError, load_ruasd_artifact_catalog
from kds.data.v4_selection import (
    V4SelectionError,
    load_v4_exposure_inventory,
    load_v4_selection_config,
    publish_v4_train_candidate_selection,
    select_v4_ksc2_candidates,
    select_v4_ruasd_candidates,
)


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4SelectionError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise V4SelectionError(f"{label} must be a JSON object.")
    return value


def _progress(completed: int, total: int, name: str) -> None:
    print(
        json.dumps(
            {
                "status": "progress",
                "stage": "source_scan",
                "completed": completed,
                "total": total,
                "item": name,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--ruasd-archive-dir", type=Path, required=True)
    parser.add_argument("--ruasd-catalog", type=Path, required=True)
    parser.add_argument("--ruasd-audit", type=Path, required=True)
    parser.add_argument("--ksc2-parts-directory", type=Path, required=True)
    parser.add_argument("--ksc2-audit", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        config = load_v4_selection_config(arguments.config)
        capacity_path = arguments.project_root / config.capacity_receipt_path
        if sha256_file(capacity_path) != config.capacity_receipt_sha256:
            raise V4SelectionError("Frozen capacity receipt SHA-256 mismatch.")
        capacity = _json_object(capacity_path, "capacity receipt")
        if capacity.get("decision") != "proceed_24k":
            raise V4SelectionError("Capacity receipt does not authorize 24k candidate selection.")
        ruasd_audit = _json_object(arguments.ruasd_audit, "RuASD audit")
        if (
            ruasd_audit.get("archive_count") != 250
            or ruasd_audit.get("sha256_verified_archives") != 250
        ):
            raise V4SelectionError("RuASD selection is not bound to a complete exact audit.")
        ksc2_audit = _json_object(arguments.ksc2_audit, "KSC2 audit")
        compressed_hash = ksc2_audit.get("compressed_sha256")
        if not isinstance(compressed_hash, str):
            raise V4SelectionError("KSC2 audit has no combined archive SHA-256.")
        exposure = load_v4_exposure_inventory(
            arguments.manifest_root, arguments.project_root
        )
        ruasd_selection = select_v4_ruasd_candidates(
            arguments.ruasd_archive_dir,
            load_ruasd_artifact_catalog(arguments.ruasd_catalog),
            config=config,
            exposure=exposure,
            progress_callback=_progress,
        )
        ksc2_candidates = scan_ksc2_text_candidates(
            arguments.ksc2_parts_directory,
            allowed_components=frozenset(config.ksc2_component_quotas),
            expected_compressed_sha256=compressed_hash,
            progress_callback=_progress,
        )
        ksc2_selection = select_v4_ksc2_candidates(
            ksc2_candidates,
            config=config,
            exposure=exposure,
        )
        rows = sorted(
            (*ruasd_selection.rows, *ksc2_selection.rows),
            key=lambda row: (row.language, row.label, row.selection_rank, row.candidate_id),
        )
        publish_v4_train_candidate_selection(
            output_csv=arguments.output_csv,
            output_receipt=arguments.output_receipt,
            rows=rows,
            config_path=arguments.config,
            config=config,
            exposure=exposure,
            ruasd_selection=ruasd_selection,
            ksc2_selection=ksc2_selection,
            created_at=arguments.created_at,
            source_bindings={
                "ruasd_catalog": {
                    "path": arguments.ruasd_catalog.as_posix(),
                    "sha256": sha256_file(arguments.ruasd_catalog),
                },
                "ruasd_audit": {
                    "path": arguments.ruasd_audit.as_posix(),
                    "sha256": sha256_file(arguments.ruasd_audit),
                },
                "ksc2_audit": {
                    "path": arguments.ksc2_audit.as_posix(),
                    "sha256": sha256_file(arguments.ksc2_audit),
                    "compressed_sha256": compressed_hash,
                },
            },
        )
    except (V4SelectionError, RuAsdCatalogError, Ksc2AuditError, OSError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(rows),
                "output_csv": str(arguments.output_csv),
                "output_csv_sha256": sha256_file(arguments.output_csv),
                "output_receipt": str(arguments.output_receipt),
                "output_receipt_sha256": sha256_file(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
