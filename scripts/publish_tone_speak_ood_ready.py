#!/usr/bin/env python3
"""Bind ToneSpeak raw selection, preprocessing accounting, and ready assets in one receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.tone_speak import TONE_SPEAK_SOURCE_ID, ToneSpeakAuditError


def _validate_raw_and_ready(raw: list[ManifestRow], ready: list[ManifestRow]) -> None:
    if len(raw) != 100 or any(
        row.split != "ood"
        or row.label != "spoof"
        or row.language != "ru"
        or row.source_name != TONE_SPEAK_SOURCE_ID
        for row in raw
    ):
        raise ToneSpeakAuditError(
            "ToneSpeak raw candidate must contain exactly 100 RU spoof OOD rows."
        )
    raw_by_id = {row.sample_id: row for row in raw}
    ready_by_id = {row.sample_id: row for row in ready}
    if len(raw_by_id) != len(raw) or len(ready_by_id) != len(ready):
        raise ToneSpeakAuditError("ToneSpeak raw or ready manifest has duplicate sample IDs.")
    for sample_id, row in ready_by_id.items():
        original = raw_by_id.get(sample_id)
        if original is None:
            raise ToneSpeakAuditError(f"ToneSpeak ready row has no raw source row: {sample_id!r}.")
        if (
            row.split != "ood"
            or row.label != "spoof"
            or row.language != "ru"
            or row.source_name != TONE_SPEAK_SOURCE_ID
            or row.text_hash != original.text_hash
            or row.text_id != original.text_id
            or row.voice_id != original.voice_id
            or row.generator_name != original.generator_name
            or row.codec != "wav"
        ):
            raise ToneSpeakAuditError(f"ToneSpeak ready provenance changed for {sample_id!r}.")


def _rejection_ids(path: Path, raw_path: Path) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToneSpeakAuditError(
            f"Cannot read ToneSpeak preprocessing rejection report: {path}"
        ) from error
    if payload.get("input_manifest") != str(raw_path) or not isinstance(
        payload.get("rejected_rows"), list
    ):
        raise ToneSpeakAuditError(
            "ToneSpeak preprocessing rejection report does not bind the raw manifest."
        )
    rejected = cast(list[object], payload["rejected_rows"])
    identifiers: set[str] = set()
    for item in rejected:
        if not isinstance(item, dict):
            raise ToneSpeakAuditError(
                "ToneSpeak preprocessing rejection report has invalid or duplicate IDs."
            )
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str):
            raise ToneSpeakAuditError(
                "ToneSpeak preprocessing rejection report has invalid or duplicate IDs."
            )
        identifiers.add(sample_id)
    if len(identifiers) != len(rejected):
        raise ToneSpeakAuditError(
            "ToneSpeak preprocessing rejection report has invalid or duplicate IDs."
        )
    return identifiers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--rejection-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        if arguments.output_receipt.exists() or not arguments.output_receipt.parent.is_dir():
            raise ValueError("ToneSpeak ready receipt must be new below an existing directory.")
        raw = load_manifest(arguments.raw_manifest)
        ready = load_manifest(arguments.ready_manifest)
        validate_manifest(raw)
        validate_manifest(ready)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(raw, ledger)
        validate_manifest_licenses(ready, ledger)
        require_valid_assets(raw, arguments.data_root)
        require_valid_assets(ready, arguments.data_root)
        _validate_raw_and_ready(raw, ready)
        rejected = _rejection_ids(arguments.rejection_report, arguments.raw_manifest)
        raw_ids = {row.sample_id for row in raw}
        ready_ids = {row.sample_id for row in ready}
        if rejected.difference(raw_ids) or raw_ids != ready_ids.union(rejected):
            raise ToneSpeakAuditError(
                "ToneSpeak preprocessing accounting does not partition the raw candidate."
            )
        receipt = {
            "schema_version": 1,
            "source_id": TONE_SPEAK_SOURCE_ID,
            "raw_manifest": str(arguments.raw_manifest),
            "raw_manifest_sha256": sha256_file(arguments.raw_manifest),
            "ready_manifest": str(arguments.ready_manifest),
            "ready_manifest_sha256": sha256_file(arguments.ready_manifest),
            "rejection_report": str(arguments.rejection_report),
            "rejection_report_sha256": sha256_file(arguments.rejection_report),
            "raw_rows": len(raw),
            "ready_rows": len(ready),
            "rejected_rows": len(rejected),
            "ready_voice_counts": {
                voice_id: sum(row.voice_id == voice_id for row in ready)
                for voice_id in sorted({row.voice_id for row in ready})
            },
                "rule": (
                    "The ready manifest may contain only normalized descendants of the locked raw "
                    "ToneSpeak OOD candidate; every omitted raw sample must appear in its "
                    "rejection report."
            ),
            "acoustic_language_preservation": "pending_two_independent_reviews_per_ready_asset",
            "final_or_product_eligible": False,
        }
        with arguments.output_receipt.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (LicenseLedgerError, ManifestError, ToneSpeakAuditError, OSError, ValueError) as error:
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
                "ready_rows": len(ready),
                "rejected_rows": len(rejected),
                "output_receipt": str(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
