from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.licenses import (
    load_license_ledger,
    validate_manifest_licenses,
    validate_training_protocol,
)
from kds.data.manifest import ManifestError, validate_manifest, write_manifest
from kds.data.ruasd_catalog import RuAsdCatalogError, load_ruasd_artifact_catalog
from kds.data.ruasd_research import (
    ExtractedRuAsdResearchAsset,
    RuAsdResearchError,
    extract_ruasd_research_slice,
    inspect_extracted_ruasd_research_audio,
    ruasd_research_manifest_rows,
    select_ruasd_research_records,
)
from kds.data.split import GroupSplitter, SplitConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a balanced, raw-only personal-research RuASD binary slice."
    )
    parser.add_argument("--archive-dir", type=Path, default=Path("/home/ruslan/Downloads/RuASD"))
    parser.add_argument(
        "--catalog", type=Path, default=Path("data/licenses/ruasd_v1_artifact_catalog.csv")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--license-ledger", type=Path, default=Path("data/licenses/license_ledger.csv")
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--limit-per-label", type=int, default=1_000)
    parser.add_argument("--min-per-stratum", type=int, default=1)
    parser.add_argument("--seed", default="20260809")
    parser.add_argument("--created-at", default=None)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip a fresh full-collection SHA-256 pass only after a documented pinned audit.",
    )
    arguments = parser.parse_args()

    if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    if arguments.output_manifest.exists() or not arguments.data_root.is_dir():
        raise ValueError("Output manifest already exists or data-root does not exist.")
    try:
        catalog = load_ruasd_artifact_catalog(arguments.catalog)
        selection = select_ruasd_research_records(
            arguments.archive_dir,
            catalog,
            limit_per_label=arguments.limit_per_label,
            min_per_stratum=arguments.min_per_stratum,
            seed=arguments.seed,
            verify_sha256=not arguments.skip_sha256,
        )
        destination = (
            arguments.data_root / "raw" / "ruasd_ru_v1_full" / "slices" / arguments.slice_name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_ruasd_research_slice(
            arguments.archive_dir, catalog, selection.records, destination
        )
        assets: dict[str, ExtractedRuAsdResearchAsset] = {}
        for record_key, path in extracted.items():
            duration_s, original_sr = inspect_extracted_ruasd_research_audio(path)
            assets[record_key] = ExtractedRuAsdResearchAsset(
                record_key=record_key,
                relative_path=path.relative_to(arguments.data_root).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration_s,
                original_sr=original_sr,
            )
        source_rows = ruasd_research_manifest_rows(
            selection.records, assets, created_at=arguments.created_at
        )
        rows = GroupSplitter(SplitConfig(seed=arguments.seed)).assign_rows(source_rows)
        validate_manifest(rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(rows, ledger)
        protocol = validate_training_protocol(rows, ledger, purpose="research")
        write_manifest(arguments.output_manifest, rows)
    except (RuAsdCatalogError, RuAsdResearchError, ManifestError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    split_counts = {
        split: sum(row.split == split for row in rows) for split in ("train", "dev", "test")
    }
    label_counts = {
        label: sum(row.label == label for row in rows) for label in ("bonafide", "spoof")
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(rows),
                "split_counts": split_counts,
                "label_counts": label_counts,
                "selected_stratum_counts": selection.selected_stratum_counts,
                "sha256_verified_archives": selection.sha256_verified_archives,
                "protocol": {
                    "purpose": protocol.purpose,
                    "split_counts": protocol.split_counts,
                    "source_ids": protocol.source_ids,
                },
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
