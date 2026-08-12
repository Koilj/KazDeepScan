"""Publish a write-once inventory for fresh RU/KK/mixed research-suite sources."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.fleurs import FLEURS_REVISION, FleursIngestionError, inspect_fleurs_release
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.eval.fresh_suite_inventory import (
    FreshSuiteInventoryError,
    audit_fleurs_locale_inventory,
    audit_ksc2_mixed_inventory,
)


def _read_csv(path: Path) -> list[Mapping[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise FreshSuiteInventoryError(f"Inventory CSV has no header: {path}")
            rows: list[Mapping[str, str]] = []
            for row_number, raw_row in enumerate(reader, start=2):
                if None in raw_row or any(value is None for value in raw_row.values()):
                    raise FreshSuiteInventoryError(
                        f"Inventory CSV {path} row {row_number} does not match its header."
                    )
                rows.append(cast(Mapping[str, str], raw_row))
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise FreshSuiteInventoryError(f"Cannot read inventory CSV {path}: {error}") from error


def _load_manifests(paths: list[Path]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for path in paths:
        manifest_rows = load_manifest(path)
        validate_manifest(manifest_rows)
        rows.extend(manifest_rows)
    return rows


def _input_binding(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-audit the pinned FLEURS release and publish exact fresh RU/KK/mixed capacity."
        )
    )
    parser.add_argument("--fleurs-release-root", type=Path, required=True)
    parser.add_argument("--ru-ready-manifest", type=Path, required=True)
    parser.add_argument("--ru-exposed-manifest", type=Path, action="append", required=True)
    parser.add_argument("--kk-ready-manifest", type=Path, required=True)
    parser.add_argument("--kk-exposed-manifest", type=Path, action="append", required=True)
    parser.add_argument("--mixed-candidate-csv", type=Path, required=True)
    parser.add_argument("--mixed-reviewed-csv", type=Path, action="append", required=True)
    parser.add_argument("--mixed-ready-manifest", type=Path, action="append", required=True)
    parser.add_argument("--mixed-exposed-manifest", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument(
        "--protocol-id", default="fresh-research-suite-source-inventory-v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        if not arguments.protocol_id.replace("-", "").replace("_", "").isalnum():
            raise FreshSuiteInventoryError(
                "Fresh-suite protocol-id may contain only letters, numbers, hyphens and "
                "underscores."
            )
        if arguments.output.exists():
            raise FreshSuiteInventoryError(
                f"Refusing to overwrite fresh-suite inventory: {arguments.output}"
            )
        if not arguments.output.parent.is_dir():
            raise FreshSuiteInventoryError(
                f"Fresh-suite inventory parent does not exist: {arguments.output.parent}"
            )

        ru_report, ru_records = inspect_fleurs_release(arguments.fleurs_release_root, "ru_ru")
        kk_report, kk_records = inspect_fleurs_release(arguments.fleurs_release_root, "kk_kz")
        ru_ready = load_manifest(arguments.ru_ready_manifest)
        validate_manifest(ru_ready)
        kk_ready = load_manifest(arguments.kk_ready_manifest)
        validate_manifest(kk_ready)
        ru_exposed = _load_manifests(arguments.ru_exposed_manifest)
        kk_exposed = _load_manifests(arguments.kk_exposed_manifest)
        mixed_ready = _load_manifests(arguments.mixed_ready_manifest)
        mixed_exposed = load_manifest(arguments.mixed_exposed_manifest)
        validate_manifest(mixed_exposed)

        ru_inventory = audit_fleurs_locale_inventory(
            locale="ru_ru",
            test_records=ru_records["test"],
            ready_rows=ru_ready,
            exposed_rows=ru_exposed,
        )
        kk_inventory = audit_fleurs_locale_inventory(
            locale="kk_kz",
            test_records=kk_records["test"],
            ready_rows=kk_ready,
            exposed_rows=kk_exposed,
        )
        mixed_inventory = audit_ksc2_mixed_inventory(
            candidate_rows=_read_csv(arguments.mixed_candidate_csv),
            reviewed_rows=[
                row for path in arguments.mixed_reviewed_csv for row in _read_csv(path)
            ],
            ready_rows=mixed_ready,
            exposed_rows=mixed_exposed,
        )

        input_paths = [
            arguments.ru_ready_manifest,
            *arguments.ru_exposed_manifest,
            arguments.kk_ready_manifest,
            *arguments.kk_exposed_manifest,
            arguments.mixed_candidate_csv,
            *arguments.mixed_reviewed_csv,
            *arguments.mixed_ready_manifest,
            arguments.mixed_exposed_manifest,
        ]
        payload = {
            "schema_version": 1,
            "protocol_id": arguments.protocol_id,
            "audited_at": arguments.audited_at,
            "fleurs_revision": FLEURS_REVISION,
            "fleurs_release_artifacts": {
                "ru_ru": dict(sorted(ru_report.artifacts.items())),
                "kk_kz": dict(sorted(kk_report.artifacts.items())),
            },
            "inputs": [_input_binding(path) for path in input_paths],
            "inventory": {
                "ru": ru_inventory,
                "kk": kk_inventory,
                "mixed": mixed_inventory,
            },
            "interpretation": {
                "asset_level_blind_only": True,
                "source_independent": False,
                "speaker_independent": False,
                "rule": (
                    "Fresh release counts are pre-extraction/pre-QA capacity, not approved final "
                    "assets. Mixed rows outside semantic review remain unknown and unusable."
                ),
            },
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except (
        FreshSuiteInventoryError,
        FleursIngestionError,
        ManifestError,
        OSError,
        ValueError,
    ) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "inventory": payload["inventory"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
