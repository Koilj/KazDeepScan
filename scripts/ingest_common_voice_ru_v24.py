from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_NAME,
    CommonVoiceIngestionError,
    ExtractedCommonVoiceAsset,
    common_voice_manifest_rows,
    extract_common_voice_audio_slice,
    inspect_common_voice_archive,
    inspect_extracted_common_voice_audio,
    load_common_voice_metadata_from_archive,
    select_common_voice_records,
)
from kds.data.manifest import ManifestError, validate_manifest, write_manifest
from kds.data.split import GroupSplitter, SplitConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a verified Russian Common Voice v24 bona-fide slice from a clean archive."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/common_voice_ru_v24") / COMMON_VOICE_RU_V24_ARCHIVE_NAME,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--source-splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--limit-per-source-split", type=int, default=250)
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
        report = inspect_common_voice_archive(arguments.archive)
        selected_records = select_common_voice_records(
            load_common_voice_metadata_from_archive(arguments.archive, arguments.source_splits),
            arguments.limit_per_source_split,
            arguments.seed,
        )
        destination = (
            arguments.data_root / "raw" / "common_voice_ru_v24" / "slices" / arguments.slice_name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_common_voice_audio_slice(
            arguments.archive,
            (record.clip_name for record in selected_records),
            destination,
        )
        assets: dict[str, ExtractedCommonVoiceAsset] = {}
        for clip_name, path in extracted.items():
            duration_s, original_sr = inspect_extracted_common_voice_audio(path)
            assets[clip_name] = ExtractedCommonVoiceAsset(
                clip_name=clip_name,
                relative_path=path.relative_to(arguments.data_root).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration_s,
                original_sr=original_sr,
            )
        source_rows = common_voice_manifest_rows(selected_records, assets, arguments.created_at)
        rows = GroupSplitter(SplitConfig(seed=arguments.seed)).assign_rows(source_rows)
        validate_manifest(rows)
        write_manifest(arguments.output_manifest, rows)
    except (CommonVoiceIngestionError, ManifestError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    counts = {split: sum(row.split == split for row in rows) for split in ("train", "dev", "test")}
    print(
        json.dumps(
            {
                "status": "ok",
                "archive_audio_files": report.audio_files,
                "archive_metadata_files": report.metadata_files,
                "rows": len(rows),
                "split_counts": counts,
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
