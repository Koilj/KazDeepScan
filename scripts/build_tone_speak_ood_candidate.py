#!/usr/bin/env python3
"""Publish a balanced, validation-only ToneSpeak raw OOD candidate without model inference."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest
from kds.data.tone_speak import (
    ToneSpeakAuditError,
    audit_tone_speak_release,
    extract_tone_speak_audio_slice,
    inspect_extracted_tone_speak_audio,
    load_tone_speak_records,
    select_tone_speak_validation_records,
    tone_speak_ood_manifest_rows,
)


def _excluded_text_hashes(
    manifests: list[Path], manifest_directories: list[Path]
) -> tuple[set[str], list[dict[str, str]], list[str]]:
    hashes: set[str] = set()
    receipts: list[dict[str, str]] = []
    skipped: list[str] = []
    discovered = {path.resolve(strict=True) for path in manifests}
    for directory in manifest_directories:
        if not directory.is_dir():
            raise ValueError(f"Manifest exclusion directory does not exist: {directory}")
        discovered.update(path.resolve(strict=True) for path in directory.glob("*.csv"))
    for path in sorted(discovered):
        try:
            rows = load_manifest(path)
        except ManifestError:
            if path.parent.resolve() in {directory.resolve() for directory in manifest_directories}:
                skipped.append(str(path))
                continue
            raise
        validate_manifest(rows)
        hashes.update(row.text_hash for row in rows)
        receipts.append({"path": str(path), "sha256": sha256_file(path)})
    return hashes, receipts, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--slice-name", default="ood_validation_v1")
    parser.add_argument("--per-voice", type=int, default=10)
    parser.add_argument("--seed", default="20260811")
    parser.add_argument("--created-at", default="2026-08-11T00:00:00Z")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-manifest-directory", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        if (
            arguments.output_manifest.exists()
            or arguments.output_receipt.exists()
            or not arguments.output_manifest.parent.is_dir()
            or not arguments.output_receipt.parent.is_dir()
            or not arguments.data_root.is_dir()
        ):
            raise ValueError("Outputs must be new and their parents plus data-root must exist.")
        if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "slice-name may contain only letters, numbers, hyphens, and underscores."
            )
        ledger = load_license_ledger(arguments.license_ledger)
        audit = audit_tone_speak_release(arguments.artifact_root)
        exclusions, exclusion_receipts, skipped_exclusions = _excluded_text_hashes(
            arguments.exclude_manifest, arguments.exclude_manifest_directory
        )
        records = load_tone_speak_records(arguments.artifact_root, source_split="validation")
        selected = select_tone_speak_validation_records(
            records,
            per_voice=arguments.per_voice,
            seed=arguments.seed,
            excluded_text_hashes=exclusions,
        )
        destination = (
            arguments.data_root / "raw" / "tone_speak_ru_v1" / "slices" / arguments.slice_name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_tone_speak_audio_slice(arguments.artifact_root, selected, destination)
        assets = {
            embedded_path: inspect_extracted_tone_speak_audio(
                path, embedded_path=embedded_path, data_root=arguments.data_root
            )
            for embedded_path, path in extracted.items()
        }
        rows = tone_speak_ood_manifest_rows(selected, assets, created_at=arguments.created_at)
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
        require_valid_assets(rows, arguments.data_root)
        staging = Path(
            tempfile.mkdtemp(
                prefix="kds-tone-speak-manifest-", dir=arguments.output_manifest.parent
            )
        )
        try:
            staged_manifest = staging / arguments.output_manifest.name
            staged_receipt = staging / arguments.output_receipt.name
            write_manifest(staged_manifest, rows)
            voice_counts = dict(sorted(Counter(row.voice_id for row in rows).items()))
            receipt = {
                "schema_version": 1,
                "source_id": audit.source_id,
                "source_revision": audit.revision,
                "source_audit": {
                    "rows_by_split": audit.rows_by_split,
                    "artifact_total_bytes": audit.artifact_total_bytes,
                },
                "selection_rule": (
                    "Validation-only; select exactly per_voice deterministic, "
                    "normalized-text-unique rows for each declared ToneSpeak built-in voice "
                    "after supplied manifest exclusions."
                ),
                "seed": arguments.seed,
                "per_voice": arguments.per_voice,
                "excluded_manifests": exclusion_receipts,
                "non_manifest_csvs_skipped": skipped_exclusions,
                "selected_rows": len(rows),
                "selected_voice_counts": voice_counts,
                "raw_manifest": str(arguments.output_manifest),
                "raw_manifest_sha256": sha256_file(staged_manifest),
                "assets": [
                    {
                        "sample_id": row.sample_id,
                        "relative_path": row.relative_path,
                        "sha256": row.sha256,
                        "text_hash": row.text_hash,
                        "voice_id": row.voice_id,
                    }
                    for row in rows
                ],
            }
            with staged_receipt.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            if arguments.output_manifest.exists() or arguments.output_receipt.exists():
                raise ValueError("A ToneSpeak output appeared while publication was staging.")
            staged_manifest.replace(arguments.output_manifest)
            staged_receipt.replace(arguments.output_receipt)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
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
                "raw_rows": len(rows),
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
