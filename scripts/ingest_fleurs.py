"""Build a text-group-disjoint bona-fide FLEURS final-candidate slice."""

from __future__ import annotations

import argparse
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.fleurs import (
    FleursExtractedAsset,
    FleursIngestionError,
    extract_fleurs_audio_slice,
    fleurs_locale_spec,
    fleurs_manifest_rows,
    inspect_extracted_fleurs_audio,
    inspect_fleurs_release,
    select_fleurs_records,
)
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest


def _existing_fleurs_exclusions(manifests: list[Path]) -> tuple[set[str], set[str]]:
    filenames: set[str] = set()
    text_hashes: set[str] = set()
    for manifest_path in manifests:
        rows = load_manifest(manifest_path)
        validate_manifest(rows)
        for row in rows:
            text_hashes.add(row.text_hash)
            if row.source_name.startswith("google_fleurs_"):
                filenames.add(f"{row.sample_id.rsplit(':', maxsplit=1)[-1]}.wav")
    return filenames, text_hashes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a complete pinned FLEURS locale and atomically publish one bona-fide "
            "recording per transcript group."
        )
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--locale", choices=("kk_kz", "ru_ru"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--source-split", choices=("test",), default="test")
    parser.add_argument("--manifest-split", choices=("test", "ood"), default="test")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", default="20260810")
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Any prior role or final manifest whose text groups may not be reused.",
    )
    parser.add_argument("--created-at", default=None)
    arguments = parser.parse_args()

    if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    if arguments.output_manifest.exists():
        raise ValueError(f"Refusing to overwrite manifest: {arguments.output_manifest}")
    if not arguments.output_manifest.parent.is_dir() or not arguments.data_root.is_dir():
        raise ValueError("FLEURS output and data-root parent directories must already exist.")
    try:
        locale_spec = fleurs_locale_spec(arguments.locale)
        _report, records_by_split = inspect_fleurs_release(arguments.release_root, arguments.locale)
        excluded_filenames, excluded_text_hashes = _existing_fleurs_exclusions(
            arguments.exclude_manifest
        )
        selected = select_fleurs_records(
            records_by_split[arguments.source_split],
            arguments.limit,
            arguments.seed,
            excluded_filenames=excluded_filenames,
            excluded_text_hashes=excluded_text_hashes,
        )
        destination = (
            arguments.data_root / "raw" / locale_spec.source_id / "slices" / arguments.slice_name
        )
        # This is a fixed, data-root-relative staging parent rather than a user-controlled
        # extraction path. The slice directory itself remains absent until atomic publication.
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_fleurs_audio_slice(
            arguments.release_root,
            arguments.locale,
            arguments.source_split,
            selected,
            destination,
        )
        assets: dict[str, FleursExtractedAsset] = {}
        for filename, path in extracted.items():
            duration_s, original_sr, codec = inspect_extracted_fleurs_audio(path)
            assets[filename] = FleursExtractedAsset(
                filename=filename,
                relative_path=path.relative_to(arguments.data_root).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration_s,
                original_sr=original_sr,
                codec=codec,
            )
        rows = fleurs_manifest_rows(
            selected,
            assets,
            manifest_split=arguments.manifest_split,
            created_at=arguments.created_at,
        )
        validate_manifest(rows)
        validate_manifest_licenses(rows, load_license_ledger(arguments.license_ledger))
        write_manifest(arguments.output_manifest, rows)
    except (FleursIngestionError, ManifestError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Published {len(rows)} FLEURS {arguments.locale} rows to {arguments.output_manifest}; "
        f"raw assets: {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
