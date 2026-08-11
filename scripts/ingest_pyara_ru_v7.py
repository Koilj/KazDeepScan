from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest
from kds.data.pyara import (
    PYARA_ARCHIVE_NAME,
    ExtractedPyAraAsset,
    PyAraIngestionError,
    extract_pyara_audio_slice,
    inspect_extracted_pyara_audio,
    inspect_pyara_archive,
    pyara_manifest_rows,
    select_pyara_records,
)
from kds.data.split import GroupSplitter, SplitConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a research-only, text-safe PyAra Russian slice."
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("/home/ruslan/Downloads") / PYARA_ARCHIVE_NAME
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--license-ledger", type=Path, default=Path("data/licenses/license_ledger.csv")
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--real-limit", type=int, default=250)
    parser.add_argument("--fake-limit-per-algorithm", type=int, default=50)
    parser.add_argument("--seed", default="20260809")
    parser.add_argument("--created-at", default=None)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Exclude every prior PyAra sample ID and text hash; may be repeated.",
    )
    parser.add_argument(
        "--fixed-split",
        choices=("train", "dev", "test"),
        default=None,
        help="Assign the whole fresh slice to one role instead of making a local three-way split.",
    )
    arguments = parser.parse_args()

    if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    if arguments.output_manifest.exists() or not arguments.data_root.is_dir():
        raise ValueError("Output manifest already exists or data-root does not exist.")
    try:
        excluded_record_ids: set[str] = set()
        excluded_text_hashes: set[str] = set()
        for manifest_path in arguments.exclude_manifest:
            excluded_rows = load_manifest(manifest_path)
            validate_manifest(excluded_rows)
            for row in excluded_rows:
                if row.source_name != "pyara_ru_v7" or not row.sample_id.startswith("pyara_ru_v7:"):
                    raise PyAraIngestionError(
                        f"Exclusion manifest is not exclusively PyAra v7: {manifest_path}"
                    )
                excluded_record_ids.add(row.sample_id.removeprefix("pyara_ru_v7:"))
                excluded_text_hashes.add(row.text_hash)
        report, records = inspect_pyara_archive(arguments.archive)
        selected = select_pyara_records(
            records,
            arguments.real_limit,
            arguments.fake_limit_per_algorithm,
            arguments.seed,
            excluded_record_ids=excluded_record_ids,
            excluded_text_hashes=excluded_text_hashes,
        )
        destination = arguments.data_root / "raw" / "pyara_ru_v7" / "slices" / arguments.slice_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_pyara_audio_slice(arguments.archive, selected, destination)
        assets: dict[str, ExtractedPyAraAsset] = {}
        for relative_path, path in extracted.items():
            duration_s, original_sr = inspect_extracted_pyara_audio(path)
            assets[relative_path] = ExtractedPyAraAsset(
                relative_path=path.relative_to(arguments.data_root).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration_s,
                original_sr=original_sr,
            )
        source_rows = pyara_manifest_rows(
            selected,
            assets,
            arguments.created_at,
            split=arguments.fixed_split or "train",
        )
        rows = (
            source_rows
            if arguments.fixed_split is not None
            else GroupSplitter(SplitConfig(seed=arguments.seed)).assign_rows(source_rows)
        )
        validate_manifest(rows)
        validate_manifest_licenses(rows, load_license_ledger(arguments.license_ledger))
        write_manifest(arguments.output_manifest, rows)
    except (LicenseLedgerError, PyAraIngestionError, ManifestError, ValueError) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (LicenseLedgerError, ManifestError))
            else [str(error)]
        )
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
                "archive_audio_files": report.audio_files,
                "archive_real_files": report.real_files,
                "archive_fake_files": report.fake_files,
                "rows": len(rows),
                "split_counts": split_counts,
                "label_counts": label_counts,
                "excluded_record_ids": len(excluded_record_ids),
                "excluded_text_hashes": len(excluded_text_hashes),
                "fixed_split": arguments.fixed_split,
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
