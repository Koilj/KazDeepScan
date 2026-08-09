from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, validate_manifest, write_manifest
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
    arguments = parser.parse_args()

    if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    if arguments.output_manifest.exists() or not arguments.data_root.is_dir():
        raise ValueError("Output manifest already exists or data-root does not exist.")
    try:
        report, records = inspect_pyara_archive(arguments.archive)
        selected = select_pyara_records(
            records, arguments.real_limit, arguments.fake_limit_per_algorithm, arguments.seed
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
        source_rows = pyara_manifest_rows(selected, assets, arguments.created_at)
        rows = GroupSplitter(SplitConfig(seed=arguments.seed)).assign_rows(source_rows)
        validate_manifest(rows)
        validate_manifest_licenses(rows, load_license_ledger(arguments.license_ledger))
        write_manifest(arguments.output_manifest, rows)
    except (PyAraIngestionError, ManifestError, ValueError) as error:
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
                "archive_audio_files": report.audio_files,
                "archive_real_files": report.real_files,
                "archive_fake_files": report.fake_files,
                "rows": len(rows),
                "split_counts": split_counts,
                "label_counts": label_counts,
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
