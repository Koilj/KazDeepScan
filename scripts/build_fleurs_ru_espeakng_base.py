"""Publish the FLEURS RU rows not already exposed by the frozen Silero candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.fleurs_espeakng import (
    FLEURS_RU_ESPEAKNG_HOLDOUT_REASON,
    FleursEspeakNgError,
    select_fleurs_ru_espeakng_base,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-base-manifest", type=Path, required=True)
    parser.add_argument("--existing-candidate-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        if arguments.output_manifest.exists() or arguments.output_receipt.exists():
            raise ValueError("Refusing to overwrite FLEURS RU eSpeak base manifest or receipt.")
        if (
            not arguments.output_manifest.parent.is_dir()
            or not arguments.output_receipt.parent.is_dir()
            or not arguments.data_root.is_dir()
        ):
            raise ValueError("Output parents and data-root must already exist.")
        full = load_manifest(arguments.full_base_manifest)
        existing = load_manifest(arguments.existing_candidate_manifest)
        validate_manifest(full)
        validate_manifest(existing)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(full, ledger)
        validate_manifest_licenses(existing, ledger)
        require_valid_assets(full, arguments.data_root)
        selected, held = select_fleurs_ru_espeakng_base(full, existing)
        staging = Path(
            tempfile.mkdtemp(
                prefix="kds-fleurs-ru-espeak-base-", dir=arguments.output_manifest.parent
            )
        )
        try:
            staged_manifest = staging / arguments.output_manifest.name
            staged_receipt = staging / arguments.output_receipt.name
            write_manifest(staged_manifest, selected)
            receipt = {
                "schema_version": 1,
                "full_base_manifest": str(arguments.full_base_manifest),
                "full_base_manifest_sha256": sha256_file(arguments.full_base_manifest),
                "existing_candidate_manifest": str(arguments.existing_candidate_manifest),
                "existing_candidate_manifest_sha256": sha256_file(
                    arguments.existing_candidate_manifest
                ),
                "selection_rule": (
                    "Select every FLEURS RU ready test bona-fide text group absent from the "
                    "existing frozen FLEURS RU/Silero candidate."
                ),
                "holdout_reason": FLEURS_RU_ESPEAKNG_HOLDOUT_REASON,
                "selected_rows": len(selected),
                "selected_manifest": str(arguments.output_manifest),
                "selected_manifest_sha256": sha256_file(staged_manifest),
                "held_out_rows": len(held),
                "held_out": [
                    {"sample_id": row.sample_id, "text_id": row.text_id, "text_hash": row.text_hash}
                    for row in held
                ],
            }
            with staged_receipt.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            if arguments.output_manifest.exists() or arguments.output_receipt.exists():
                raise ValueError(
                    "A FLEURS RU eSpeak output appeared while publication was staging."
                )
            staged_manifest.replace(arguments.output_manifest)
            staged_receipt.replace(arguments.output_receipt)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    except (FleursEspeakNgError, LicenseLedgerError, ManifestError, OSError, ValueError) as error:
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
                "selected_rows": len(selected),
                "held_out_rows": len(held),
                "output_manifest": str(arguments.output_manifest),
                "output_receipt": str(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
