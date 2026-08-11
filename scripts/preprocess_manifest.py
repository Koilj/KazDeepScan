from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import require_valid_assets
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.preprocess import preprocess_rows, reuse_preprocessed_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a verified raw manifest into ready 16 kHz mono WAV assets."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument(
        "--allow-rejections",
        action="store_true",
        help="Publish only ready assets; requires --rejection-report for every excluded row.",
    )
    parser.add_argument(
        "--rejection-report",
        type=Path,
        help="New JSON report for excluded assets; required with --allow-rejections.",
    )
    parser.add_argument("--reuse-raw-manifest", type=Path)
    parser.add_argument("--reuse-ready-manifest", type=Path)
    arguments = parser.parse_args()

    try:
        if arguments.allow_rejections != (arguments.rejection_report is not None):
            raise ValueError("--allow-rejections and --rejection-report must be used together.")
        if (arguments.reuse_raw_manifest is None) != (
            arguments.reuse_ready_manifest is None
        ):
            raise ValueError("Both reuse manifests must be supplied together.")
        rows = load_manifest(arguments.input_manifest)
        validate_manifest(rows)
        validate_manifest_licenses(rows, load_license_ledger(arguments.license_ledger))
        if arguments.output_manifest.exists():
            raise ValueError(
                f"Refusing to overwrite existing manifest: {arguments.output_manifest}"
            )
        if not arguments.output_manifest.parent.is_dir():
            raise ValueError(
                f"Manifest output directory does not exist: {arguments.output_manifest.parent}"
            )
        if arguments.rejection_report is not None:
            if arguments.rejection_report.exists():
                raise ValueError(
                    f"Refusing to overwrite rejection report: {arguments.rejection_report}"
                )
            if not arguments.rejection_report.parent.is_dir():
                raise ValueError(
                    "Rejection report directory does not exist: "
                    f"{arguments.rejection_report.parent}"
                )
        require_valid_assets(rows, arguments.data_root)
        reused_rows: tuple[ManifestRow, ...] = ()
        rows_to_process = rows
        if arguments.reuse_raw_manifest is not None:
            prior_raw_rows = load_manifest(arguments.reuse_raw_manifest)
            prior_ready_rows = load_manifest(arguments.reuse_ready_manifest)
            validate_manifest(prior_raw_rows)
            validate_manifest(prior_ready_rows)
            require_valid_assets(prior_raw_rows, arguments.data_root)
            require_valid_assets(prior_ready_rows, arguments.data_root)
            reuse = reuse_preprocessed_rows(rows, prior_raw_rows, prior_ready_rows)
            reused_rows = reuse.reused_rows
            rows_to_process = list(reuse.remaining_rows)
        report = preprocess_rows(
            rows_to_process,
            arguments.data_root,
            AudioPreparationPipeline(),
            allow_rejections=arguments.allow_rejections,
        )
        if report.issues and not arguments.allow_rejections:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "processed": len(report.processed_rows),
                        "issues": [asdict(issue) for issue in report.issues],
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        if arguments.rejection_report is not None:
            payload = {
                "input_manifest": str(arguments.input_manifest),
                "reused_rows": len(reused_rows),
                "published_rows": len(reused_rows) + len(report.processed_rows),
                "rejected_rows": [asdict(issue) for issue in report.issues],
            }
            with arguments.rejection_report.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        published_rows = sorted(
            (*reused_rows, *report.processed_rows), key=lambda row: row.sample_id
        )
        validate_manifest(published_rows)
        write_manifest(arguments.output_manifest, published_rows)
    except (LicenseLedgerError, ManifestError, ValueError) as error:
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
                "processed": len(report.processed_rows),
                "reused": len(reused_rows),
                "published": len(reused_rows) + len(report.processed_rows),
                "rejected": len(report.issues),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
