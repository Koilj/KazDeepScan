from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.ksc_slr102 import (
    KSC_ARCHIVE_NAME,
    ExtractedKscAsset,
    KscIngestionError,
    attach_ksc_transcripts,
    extract_ksc_audio_slice,
    inspect_extracted_ksc_audio,
    inspect_ksc_archive,
    ksc_manifest_rows,
    load_ksc_metadata,
    load_ksc_metadata_from_archive,
    select_ksc_records,
    select_ksc_records_from_archive_excluding_texts,
)
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest


def _excluded_ksc_provenance(manifests: list[Path]) -> tuple[set[str], set[str]]:
    """Read previously frozen KSC rows to prevent a new derived source from reusing them."""

    utterance_ids: set[str] = set()
    text_hashes: set[str] = set()
    prefix = "ksc_slr102:"
    for path in manifests:
        rows = load_manifest(path)
        validate_manifest(rows)
        for row in rows:
            if row.source_name != "ksc_slr102":
                continue
            if not row.sample_id.startswith(prefix):
                raise ValueError(f"KSC row has unexpected sample_id: {row.sample_id!r}")
            utterance_id = row.sample_id.removeprefix(prefix)
            if not utterance_id or "/" in utterance_id or "\\" in utterance_id:
                raise ValueError(f"KSC row has unsafe sample_id: {row.sample_id!r}")
            utterance_ids.add(utterance_id)
            text_hashes.add(row.text_hash)
    return utterance_ids, text_hashes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a verified bona-fide KSC slice from a clean archive and its source metadata."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/ksc_slr102") / KSC_ARCHIVE_NAME,
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Optional extracted KSC root; omit to use Meta/ and Transcriptions/ in archive.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--source-splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--limit-per-split", type=int, default=250)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Existing frozen KSC manifest; its KSC sample IDs and text hashes cannot be reused.",
    )
    parser.add_argument("--seed", default="20260808")
    parser.add_argument("--created-at", default=None)
    arguments = parser.parse_args()

    if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    if arguments.output_manifest.exists():
        raise ValueError(f"Refusing to overwrite manifest: {arguments.output_manifest}")
    if not arguments.data_root.is_dir():
        raise ValueError(f"data-root does not exist: {arguments.data_root}")

    try:
        excluded_ids, excluded_text_hashes = _excluded_ksc_provenance(arguments.exclude_manifest)
        if arguments.metadata_root is None:
            selected_index_records, report = select_ksc_records_from_archive_excluding_texts(
                arguments.archive,
                load_ksc_metadata_from_archive(arguments.archive, arguments.source_splits),
                arguments.limit_per_split,
                arguments.seed,
                excluded_utterance_ids=excluded_ids,
                excluded_text_hashes=excluded_text_hashes,
            )
        else:
            report = inspect_ksc_archive(arguments.archive)
            selected_records = select_ksc_records(
                load_ksc_metadata(arguments.metadata_root, arguments.source_splits),
                arguments.limit_per_split,
                arguments.seed,
                excluded_utterance_ids=excluded_ids,
            )
        destination = arguments.data_root / "raw" / "ksc_slr102" / "slices" / arguments.slice_name
        selected_ids = (
            (record.utterance_id for record in selected_index_records)
            if arguments.metadata_root is None
            else (record.utterance_id for record in selected_records)
        )
        extracted = extract_ksc_audio_slice(
            arguments.archive,
            selected_ids,
            destination,
            excluded_text_hashes=excluded_text_hashes,
        )
        if arguments.metadata_root is None:
            selected_records = attach_ksc_transcripts(selected_index_records, destination)
        assets: dict[str, ExtractedKscAsset] = {}
        for utterance_id, path in extracted.items():
            duration_s, original_sr, codec = inspect_extracted_ksc_audio(path)
            assets[utterance_id] = ExtractedKscAsset(
                utterance_id=utterance_id,
                relative_path=path.relative_to(arguments.data_root).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration_s,
                original_sr=original_sr,
                codec=codec,
            )
        rows = ksc_manifest_rows(selected_records, assets, arguments.created_at)
        reused_text_hashes = sorted(
            {row.text_hash for row in rows}.intersection(excluded_text_hashes)
        )
        if reused_text_hashes:
            raise ValueError(
                "KSC selection overlaps a frozen manifest by transcript text hash; "
                f"found {len(reused_text_hashes)} collisions."
            )
        validate_manifest(rows)
        write_manifest(arguments.output_manifest, rows)
    except (KscIngestionError, ManifestError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    counts = {split: sum(row.split == split for row in rows) for split in ("train", "dev", "test")}
    print(
        json.dumps(
            {
                "status": "ok",
                "archive_audio_files": report.audio_files,
                "archive_transcript_files": report.transcript_files,
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
