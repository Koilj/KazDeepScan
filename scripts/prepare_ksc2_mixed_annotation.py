"""Publish an unlabelled KSC2 priority-component packet for later mixed review.

This is intentionally not a KDS training manifest. It contains only pending
candidates and has no option that can assign ``language=mixed`` or
``code_switch=true`` automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kds.data.ksc2 import (
    KSC2_MIXED_ANNOTATION_COMPONENTS,
    KSC2_PART_EXPECTED_SIZES,
    Ksc2AnnotationCandidate,
    Ksc2AuditError,
    extract_ksc2_mixed_annotation_candidates,
)
from kds.data.ksc2_mixed_review import KSC2_MIXED_CANDIDATE_FIELDS

ANNOTATION_FIELDS = KSC2_MIXED_CANDIDATE_FIELDS


def _slice_name(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    return value


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"KSC2 source lock {label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def load_ksc2_annotation_lock(path: Path) -> tuple[str, str, str]:
    """Return the pinned archive hash, declared license and lock-file hash."""

    payload = path.read_bytes()
    lock_hash = hashlib.sha256(payload).hexdigest()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot parse KSC2 source lock: {error}") from error
    lock = _object(value, "root")
    if lock.get("source_id") != "ksc2_v1":
        raise ValueError("KSC2 source lock source_id must be 'ksc2_v1'.")
    license_value = lock.get("license")
    archive = _object(lock.get("multipart_archive"), "multipart_archive")
    compressed_hash = archive.get("compressed_sha256")
    parts = archive.get("parts")
    if (
        not isinstance(license_value, str)
        or not isinstance(compressed_hash, str)
        or not isinstance(parts, list)
    ):
        raise ValueError("KSC2 source lock has incomplete license or multipart archive fields.")
    try:
        lock_sizes = tuple(_object(part, "multipart_archive.parts")["size_bytes"] for part in parts)
    except KeyError as error:
        raise ValueError("KSC2 source lock part lacks size_bytes.") from error
    if lock_sizes != KSC2_PART_EXPECTED_SIZES:
        raise ValueError("KSC2 source lock part-size contract differs from the audited release.")
    return compressed_hash, license_value, lock_hash


def annotation_rows(
    candidates: list[Ksc2AnnotationCandidate],
    *,
    slice_name: str,
    archive_sha256: str,
    source_license: str,
    source_lock_sha256: str,
) -> list[dict[str, str]]:
    """Render immutable candidates as pending rows without a mixed decision."""

    prefix = Path("raw") / "ksc2_v1" / "slices" / slice_name
    return [
        {
            "annotation_id": f"ksc2_v1:{candidate.candidate_id}",
            "component": candidate.component,
            "audio_relative_path": (prefix / candidate.audio_relative_path).as_posix(),
            "audio_sha256": candidate.audio_sha256,
            "archive_audio_member": candidate.archive_audio_member,
            "archive_transcript_member": candidate.archive_transcript_member,
            "transcript": candidate.transcript,
            "transcript_sha256": candidate.transcript_sha256,
            "source_name": "ksc2_v1",
            "source_license": source_license,
            "archive_sha256": archive_sha256,
            "source_lock_sha256": source_lock_sha256,
            "annotation_state": "pending",
            "language": "unknown",
            "code_switch": "unknown",
        }
        for candidate in candidates
    ]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only pending KSC2 Test/podcasts, Test/talkshow and Test/radio candidates "
            "for later review. It never emits a mixed-label manifest."
        )
    )
    parser.add_argument("--parts-directory", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        slice_name = _slice_name(arguments.slice_name)
        if arguments.output_csv.exists() or arguments.receipt.exists():
            raise ValueError("Refusing to overwrite KSC2 annotation CSV or receipt.")
        if not arguments.output_csv.parent.is_dir() or not arguments.receipt.parent.is_dir():
            raise ValueError("KSC2 annotation CSV and receipt parents must already exist.")
        data_root = arguments.data_root.resolve(strict=True)
        final_slice = data_root / "raw" / "ksc2_v1" / "slices" / slice_name
        if final_slice.exists():
            raise ValueError(f"Refusing to overwrite KSC2 annotation slice: {final_slice}")
        final_slice.parent.mkdir(parents=True, exist_ok=True)
        expected_hash, source_license, lock_hash = load_ksc2_annotation_lock(arguments.source_lock)

        stage_root = Path(tempfile.mkdtemp(prefix="kds-ksc2-annotation-", dir=final_slice.parent))
        csv_stage = Path(
            tempfile.mkdtemp(prefix="kds-ksc2-annotation-csv-", dir=arguments.output_csv.parent)
        )
        receipt_stage = Path(
            tempfile.mkdtemp(prefix="kds-ksc2-annotation-receipt-", dir=arguments.receipt.parent)
        )
        try:
            staged_slice = stage_root / "slice"
            candidates = list(
                extract_ksc2_mixed_annotation_candidates(
                    arguments.parts_directory,
                    staged_slice,
                    expected_compressed_sha256=expected_hash,
                )
            )
            rows = annotation_rows(
                candidates,
                slice_name=slice_name,
                archive_sha256=expected_hash,
                source_license=source_license,
                source_lock_sha256=lock_hash,
            )
            component_counts = Counter(row["component"] for row in rows)
            if set(component_counts) != set(KSC2_MIXED_ANNOTATION_COMPONENTS):
                raise ValueError(
                    "KSC2 annotation extraction did not cover every priority component."
                )
            _write_rows(csv_stage / arguments.output_csv.name, rows)
            (receipt_stage / arguments.receipt.name).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_lock": str(arguments.source_lock),
                        "source_lock_sha256": lock_hash,
                        "archive_sha256": expected_hash,
                        "priority_components": sorted(KSC2_MIXED_ANNOTATION_COMPONENTS),
                        "candidate_count": len(rows),
                        "candidate_counts_by_component": dict(sorted(component_counts.items())),
                        "annotation_state": "pending",
                        "language": "unknown",
                        "code_switch": "unknown",
                        "rule": (
                            "No automatic language decision. The packet is an immutable candidate "
                            "source; any later review must preserve explicit per-row evidence."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.move(str(staged_slice), str(final_slice))
            shutil.move(str(csv_stage / arguments.output_csv.name), str(arguments.output_csv))
            shutil.move(str(receipt_stage / arguments.receipt.name), str(arguments.receipt))
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
            shutil.rmtree(csv_stage, ignore_errors=True)
            shutil.rmtree(receipt_stage, ignore_errors=True)
    except (Ksc2AuditError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(rows),
                "components": dict(sorted(component_counts.items())),
                "output_csv": str(arguments.output_csv),
                "receipt": str(arguments.receipt),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
