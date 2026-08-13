"""Write a source-wide Denis exact-identity and historical-lineage exposure receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.denis import (
    DENIS_ARCHIVE_EXPECTED_SHA256,
    DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
    DENIS_SOURCE_ID,
    DenisArchiveAuditError,
    inspect_denis_archive,
)
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.denis_source_screen import screen_denis_source_records


def _load_source_audit(path: Path) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DenisArchiveAuditError(f"Cannot read Denis source audit receipt: {path}.") from error
    if not isinstance(payload, dict):
        raise DenisArchiveAuditError("Denis source audit receipt must be a JSON object.")
    required: dict[str, object] = {
        "source_id": DENIS_SOURCE_ID,
        "archive_size_bytes": DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
        "archive_sha256": DENIS_ARCHIVE_EXPECTED_SHA256,
        "paired_records": 1150,
        "disk_extraction_performed": False,
        "candidate_selection_performed": False,
        "tts_inference_performed": False,
        "detector_inference_performed": False,
        "intake_status": "accepted_source_level_only",
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise DenisArchiveAuditError(
                f"Denis source audit receipt has unexpected {field!r}: {payload.get(field)!r}."
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
        inspection = inspect_denis_archive(arguments.archive)
        if (
            inspection.audit.paired_records != source_audit["paired_records"]
            or inspection.audit.record_identity_fingerprint
            != source_audit.get("record_identity_fingerprint")
        ):
            raise DenisArchiveAuditError(
                "Denis archive inspection differs from the source audit receipt."
            )
        payload = screen_denis_source_records(
            records=inspection.records,
            project_root=arguments.project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
            created_at=arguments.created_at,
            source_audit_receipt={
                "path": str(arguments.source_audit_receipt),
                "sha256": sha256_file(arguments.source_audit_receipt),
            },
        )
        with arguments.output.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
    except (CandidateExposureError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    claims = cast(dict[str, object], payload["claims"])
    lineage = cast(dict[str, object], payload["historical_likely_speaker_lineage"])
    configured = cast(dict[str, object], lineage["configured_scope"])
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "exact_source_absent": claims[
                    "exact_source_sample_audio_and_text_absent_from_historical_project_scope"
                ],
                "historical_denis_unique_sample_ids": configured["unique_sample_ids"],
                "speaker_independent": claims["speaker_independent"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
