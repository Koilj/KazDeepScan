"""Publish a write-once pre-extraction project-exposure screen for VoxForge Russian."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.voxforge import (
    VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
    VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
    VOXFORGE_RU_SOURCE_ID,
    VoxForgeRuAuditError,
    load_voxforge_ru_metadata,
)
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.voxforge_metadata_screen import screen_voxforge_ru_metadata


def _load_source_audit(path: Path) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgeRuAuditError(f"Cannot read VoxForge source audit receipt: {path}.") from error
    if not isinstance(payload, dict):
        raise VoxForgeRuAuditError("VoxForge source audit receipt must be a JSON object.")
    required = {
        "source_id": VOXFORGE_RU_SOURCE_ID,
        "archive_sha256": VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
        "archive_size_bytes": VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
        "submissions": 644,
        "wav_files": 6412,
        "prompt_rows": 6412,
        "original_prompt_rows": 6412,
        "extraction_performed": False,
        "candidate_selection_performed": False,
        "detector_inference_performed": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise VoxForgeRuAuditError(
                f"VoxForge source audit receipt has unexpected {field!r}: {payload.get(field)!r}."
            )
    return cast(dict[str, object], payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-audit-receipt", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise CandidateExposureError("Screen output must be new with an existing parent.")
        source_audit = _load_source_audit(arguments.source_audit_receipt)
        records = load_voxforge_ru_metadata(arguments.archive)
        if len(records) != source_audit["wav_files"]:
            raise VoxForgeRuAuditError("VoxForge metadata count differs from source audit receipt.")
        screen = screen_voxforge_ru_metadata(
            records=records,
            project_root=arguments.project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
            created_at=arguments.created_at,
        )
        payload = {
            **screen.receipt,
            "archive": {
                "source_audit_receipt": {
                    "path": str(arguments.source_audit_receipt),
                    "sha256": sha256_file(arguments.source_audit_receipt),
                },
                "expected_size_bytes": VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
                "expected_sha256": VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
                "identity_verified_before_metadata_read": True,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
    except (CandidateExposureError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    strict = cast(dict[str, object], payload["strict_group_exclusion"])
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "surviving_records": strict["surviving_records"],
                "surviving_contributor_groups": strict["surviving_contributor_groups"],
                "surviving_prompt_text_groups": strict["surviving_canonical_prompt_text_groups"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
