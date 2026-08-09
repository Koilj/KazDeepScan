from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError, validate_manifest, write_manifest
from kds.data.ml_df import (
    ML_DF_IT_ARCHIVE_NAME,
    ML_DF_METADATA_ARCHIVE_NAME,
    ExtractedMlDfAsset,
    MlDfIngestionError,
    extract_ml_df_audio_slice,
    inspect_extracted_ml_df_audio,
    inspect_ml_df_archive,
    load_ml_df_it_metadata,
    ml_df_ood_manifest_rows,
    select_ml_df_ood_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a verified ML-DF Italian OOD audio slice.")
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/ml_df") / ML_DF_IT_ARCHIVE_NAME
    )
    parser.add_argument(
        "--metadata-archive",
        type=Path,
        default=Path("data/raw/ml_df") / ML_DF_METADATA_ARCHIVE_NAME,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--bonafide-limit", type=int, default=100)
    parser.add_argument("--spoof-limit-per-generator", type=int, default=25)
    parser.add_argument("--seed", default="20260809")
    parser.add_argument("--created-at", default=None)
    arguments = parser.parse_args()

    if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    if arguments.output_manifest.exists():
        raise ValueError(f"Refusing to overwrite manifest: {arguments.output_manifest}")
    if not arguments.data_root.is_dir():
        raise ValueError(f"data-root does not exist: {arguments.data_root}")

    try:
        metadata = load_ml_df_it_metadata(arguments.metadata_archive)
        report = inspect_ml_df_archive(arguments.archive, metadata)
        selected = select_ml_df_ood_records(
            metadata,
            bonafide_limit=arguments.bonafide_limit,
            spoof_limit_per_generator=arguments.spoof_limit_per_generator,
            seed=arguments.seed,
        )
        destination = arguments.data_root / "raw" / "ml_df_it_v1" / "slices" / arguments.slice_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_ml_df_audio_slice(arguments.archive, selected, destination)
        destination_relative = destination.relative_to(arguments.data_root)
        assets: dict[str, ExtractedMlDfAsset] = {}
        for relative_path, path in extracted.items():
            duration_s, original_sr = inspect_extracted_ml_df_audio(path)
            assets[relative_path] = ExtractedMlDfAsset(
                relative_path=(destination_relative / relative_path).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration_s,
                original_sr=original_sr,
            )
        rows = ml_df_ood_manifest_rows(selected, assets, arguments.created_at)
        validate_manifest(rows, require_ood_generator=True)
        write_manifest(arguments.output_manifest, rows)
    except (MlDfIngestionError, ManifestError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    tools = {
        tool: sum(record.tool == tool for record in selected)
        for tool in sorted({record.tool for record in selected})
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "archive_audio_files": report.audio_files,
                "rows": len(rows),
                "tool_counts": tools,
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
