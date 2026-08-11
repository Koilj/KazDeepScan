from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import require_valid_assets
from kds.data.ksc_derived_kk import merge_prepared_rows
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge a ready slice with verified prior processed collisions."
    )
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--new-ready-manifest", type=Path, required=True)
    parser.add_argument("--reusable-ready-manifest", type=Path, required=True)
    parser.add_argument("--preprocess-rejections", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-rejections", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--source-name", default="ksc_slr102")
    parser.add_argument("--label", default="bonafide")
    parser.add_argument("--language", default="kk")
    parser.add_argument("--source-description", default="Raw KSC base")
    arguments = parser.parse_args()

    try:
        if arguments.output_manifest.exists() or arguments.output_rejections.exists():
            raise ValueError("Refusing to overwrite merged KSC manifest or rejection report.")
        if (
            not arguments.output_manifest.parent.is_dir()
            or not arguments.output_rejections.parent.is_dir()
        ):
            raise ValueError("Merged KSC output parent directory does not exist.")
        raw_rows = load_manifest(arguments.raw_manifest)
        new_rows = load_manifest(arguments.new_ready_manifest)
        reusable_rows = load_manifest(arguments.reusable_ready_manifest)
        for rows in (raw_rows, new_rows, reusable_rows):
            validate_manifest(rows)
        ledger = load_license_ledger(arguments.license_ledger)
        for rows in (raw_rows, new_rows, reusable_rows):
            validate_manifest_licenses(rows, ledger)
        merged_rows, reused_ids = merge_prepared_rows(
            raw_rows,
            new_rows,
            reusable_rows,
            source_name=arguments.source_name,
            label=arguments.label,
            language=arguments.language,
            source_description=arguments.source_description,
        )
        validate_manifest(merged_rows)
        validate_manifest_licenses(merged_rows, ledger)
        require_valid_assets(merged_rows, arguments.data_root)
        try:
            preprocessed_rejections = json.loads(
                arguments.preprocess_rejections.read_text(encoding="utf-8")
            )
            original_rejected_rows = preprocessed_rejections["rejected_rows"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot read preprocess rejection report: {error}") from error
        if not isinstance(original_rejected_rows, list):
            raise ValueError("Preprocess rejection report rejected_rows must be a JSON array.")
        if any(
            not isinstance(item, dict) or not isinstance(item.get("sample_id"), str)
            for item in original_rejected_rows
        ):
            raise ValueError("Every preprocess rejection must be an object with sample_id.")
        remaining_rejections = [
            item for item in original_rejected_rows if item["sample_id"] not in reused_ids
        ]
        ready_ids = {row.sample_id for row in merged_rows}
        raw_ids = {row.sample_id for row in raw_rows}
        rejected_ids = {item["sample_id"] for item in remaining_rejections}
        if raw_ids != ready_ids.union(rejected_ids) or ready_ids.intersection(rejected_ids):
            raise ValueError(
                "Merged ready rows and remaining rejections do not partition the raw KSC slice."
            )
        write_manifest(arguments.output_manifest, merged_rows)
        with arguments.output_rejections.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "raw_manifest": str(arguments.raw_manifest),
                    "new_ready_manifest": str(arguments.new_ready_manifest),
                    "reusable_ready_manifest": str(arguments.reusable_ready_manifest),
                    "published_rows": len(merged_rows),
                    "reused_preprocessed_rows": len(reused_ids),
                    "rejected_rows": remaining_rejections,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
    except (LicenseLedgerError, ManifestError, OSError, ValueError) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (LicenseLedgerError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "published_rows": len(merged_rows),
                "reused_preprocessed_rows": len(reused_ids),
                "rejected_rows": len(remaining_rejections),
                "manifest": str(arguments.output_manifest),
                "rejections": str(arguments.output_rejections),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
