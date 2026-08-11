"""Freeze a calibration-only dev role after excluding every prior Stage-B observation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import load_manifest, validate_manifest, write_manifest
from kds.data.stage_b_dev import filter_stage_b_calibration_rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a calibration-only Stage-B dev manifest that is leakage-safe against every "
            "earlier training and epoch-selection role."
        )
    )
    parser.add_argument(
        "--historical-manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "Previously used Stage-A/Stage-B manifest. May be repeated; all rows are exclusion "
            "keys regardless of their split."
        ),
    )
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.output_manifest.exists() or arguments.report.exists():
        raise ValueError("Calibration output manifest and report must both be new.")
    if not arguments.output_manifest.parent.is_dir() or not arguments.report.parent.is_dir():
        raise ValueError("Calibration output directories must already exist.")

    historical_rows = []
    historical_receipts = []
    for path in arguments.historical_manifest:
        rows = load_manifest(path)
        validate_manifest(rows)
        historical_rows.extend(rows)
        historical_receipts.append(
            {"path": str(path), "sha256": _sha256_file(path), "rows": len(rows)}
        )
    candidate_rows = load_manifest(arguments.candidate_manifest)
    selected, filter_report = filter_stage_b_calibration_rows(historical_rows, candidate_rows)
    ledger = load_license_ledger(arguments.license_ledger)
    validate_manifest_licenses([*historical_rows, *selected], ledger)

    write_manifest(arguments.output_manifest, selected)
    receipt = {
        "status": "ok",
        "role": "calibration_dev",
        "historical_manifests": historical_receipts,
        "candidate_manifest": {
            "path": str(arguments.candidate_manifest),
            "sha256": _sha256_file(arguments.candidate_manifest),
            "rows": len(candidate_rows),
        },
        "published_manifest": {
            "path": str(arguments.output_manifest),
            "sha256": _sha256_file(arguments.output_manifest),
            "rows": len(selected),
        },
        "filter": asdict(filter_report),
    }
    arguments.report.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
