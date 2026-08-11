from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import load_manifest, write_manifest
from kds.data.stage_b_dev import filter_stage_b_dev_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish a fresh dev manifest after removing every train leakage key."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output_manifest.exists() or arguments.report.exists():
        raise ValueError("Stage-B dev output manifest and report must both be new.")
    if not arguments.output_manifest.parent.is_dir() or not arguments.report.parent.is_dir():
        raise ValueError("Stage-B dev output directories must already exist.")

    train_rows = [row for row in load_manifest(arguments.train_manifest) if row.split == "train"]
    candidate_rows = load_manifest(arguments.candidate_manifest)
    selected, report = filter_stage_b_dev_rows(train_rows, candidate_rows)
    validate_manifest_licenses(
        [*train_rows, *selected], load_license_ledger(arguments.license_ledger)
    )
    write_manifest(arguments.output_manifest, selected)
    arguments.report.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(arguments.output_manifest),
                "report": str(arguments.report),
                **asdict(report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
