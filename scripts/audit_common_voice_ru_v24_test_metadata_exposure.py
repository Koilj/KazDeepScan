"""Publish a write-once pre-extraction Common Voice RU test metadata exposure screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
    CommonVoiceIngestionError,
    load_common_voice_metadata_from_archive,
)
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.common_voice_metadata_screen import screen_common_voice_ru_test_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise CandidateExposureError(
                "Metadata-screen output must be new with an existing parent."
            )
        records = load_common_voice_metadata_from_archive(arguments.archive, ("test",))
        screen = screen_common_voice_ru_test_metadata(
            records=records,
            project_root=arguments.project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
            created_at=arguments.audited_at,
        )
        payload = {
            **screen.receipt,
            "archive": {
                "path": str(arguments.archive),
                "expected_size_bytes": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
                "expected_sha256": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
                "identity_verified_before_metadata_read": True,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
        strict_group_exclusion = screen.receipt["strict_group_exclusion"]
        if not isinstance(strict_group_exclusion, dict):
            raise CandidateExposureError("Metadata screen has an invalid group-exclusion receipt.")
        surviving_client_groups = strict_group_exclusion.get("surviving_client_groups")
        if not isinstance(surviving_client_groups, int):
            raise CandidateExposureError("Metadata screen has no surviving-client-group count.")
    except (CandidateExposureError, CommonVoiceIngestionError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "surviving_records": len(screen.surviving),
                "surviving_client_groups": surviving_client_groups,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
