from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError, validate_manifest, write_manifest
from kds.data.ruasd import (
    RUASD_ARCHIVE_NAME,
    ExtractedRuAsdAsset,
    RuAsdIngestionError,
    extract_ruasd_ood_slice,
    inspect_extracted_ruasd_audio,
    load_ruasd_fake_records,
    ruasd_ood_manifest_rows,
    select_ruasd_ood_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a verified fake-only Russian RuASD OOD slice."
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/ruasd") / RUASD_ARCHIVE_NAME
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--limit-per-generator", type=int, default=50)
    parser.add_argument("--seed", default="20260809")
    parser.add_argument("--created-at", default=None)
    arguments = parser.parse_args()

    if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    if arguments.output_manifest.exists() or not arguments.data_root.is_dir():
        raise ValueError("Output manifest already exists or data-root does not exist.")
    try:
        records = load_ruasd_fake_records(arguments.archive)
        selected = select_ruasd_ood_records(records, arguments.limit_per_generator, arguments.seed)
        destination = arguments.data_root / "raw" / "ruasd" / "slices" / arguments.slice_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_ruasd_ood_slice(arguments.archive, selected, destination)
        assets: dict[str, ExtractedRuAsdAsset] = {}
        for sample_id, path in extracted.items():
            duration_s, original_sr = inspect_extracted_ruasd_audio(path)
            assets[sample_id] = ExtractedRuAsdAsset(
                sample_id=sample_id,
                relative_path=path.relative_to(arguments.data_root).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration_s,
                original_sr=original_sr,
            )
        rows = ruasd_ood_manifest_rows(selected, assets, arguments.created_at)
        validate_manifest(rows, require_ood_generator=True)
        write_manifest(arguments.output_manifest, rows)
    except (RuAsdIngestionError, ManifestError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    generator_names = sorted({row.generator_name for row in rows})
    generator_counts = {
        name: sum(row.generator_name == name for row in rows) for name in generator_names
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "archive_audio_files": len(records),
                "rows": len(rows),
                "generator_counts": generator_counts,
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
