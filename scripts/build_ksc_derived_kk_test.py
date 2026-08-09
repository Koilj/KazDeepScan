from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import require_valid_assets
from kds.data.ksc_derived_kk import KSC_DERIVED_KK_SOURCE_ID
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)


def _paired_bonafide_rows(
    base_rows: list[ManifestRow], spoof_rows: list[ManifestRow]
) -> list[ManifestRow]:
    bonafide_by_text = {
        row.text_id: row
        for row in base_rows
        if row.source_name == "ksc_slr102"
        and row.split == "test"
        and row.label == "bonafide"
        and row.language == "kk"
    }
    if len(bonafide_by_text) != sum(
        row.source_name == "ksc_slr102" and row.split == "test" and row.label == "bonafide"
        for row in base_rows
    ):
        raise ValueError("KSC base manifest has duplicate text_id values in test bona-fide rows.")
    paired: list[ManifestRow] = []
    seen_text_ids: set[str] = set()
    for row in spoof_rows:
        if (
            row.source_name != KSC_DERIVED_KK_SOURCE_ID
            or row.split != "test"
            or row.label != "spoof"
        ):
            raise ValueError("Spoof manifest contains a row outside the derived KSC test source.")
        base_row = bonafide_by_text.get(row.text_id)
        if base_row is None:
            raise ValueError(f"No KSC bona-fide row pairs with spoof text_id={row.text_id!r}.")
        if base_row.text_hash != row.text_hash:
            raise ValueError(f"KSC/spoof text hash mismatch for text_id={row.text_id!r}.")
        if row.text_id in seen_text_ids:
            raise ValueError(f"Multiple spoof rows pair with text_id={row.text_id!r}.")
        seen_text_ids.add(row.text_id)
        paired.append(base_row)
    if len(paired) != len(spoof_rows):
        raise ValueError("Every spoof row must have exactly one paired KSC bona-fide row.")
    return paired


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a paired KSC bona-fide + derived spoof Kazakh test manifest."
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--spoof-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        if arguments.output_manifest.exists():
            raise ValueError(
                f"Refusing to overwrite existing manifest: {arguments.output_manifest}"
            )
        base_rows = load_manifest(arguments.base_manifest)
        spoof_rows = load_manifest(arguments.spoof_manifest)
        validate_manifest(base_rows)
        validate_manifest(spoof_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        validate_manifest_licenses(spoof_rows, ledger)
        bonafide_rows = _paired_bonafide_rows(base_rows, spoof_rows)
        combined = [*bonafide_rows, *spoof_rows]
        validate_manifest(combined)
        validate_manifest_licenses(combined, ledger)
        require_valid_assets(combined, arguments.data_root)
        write_manifest(arguments.output_manifest, combined)
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
                "rows": len(combined),
                "bonafide": len(bonafide_rows),
                "spoof": len(spoof_rows),
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
