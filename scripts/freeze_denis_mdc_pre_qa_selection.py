"""Freeze the 79-row Denis category-balanced metadata selection before extraction."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.denis import (
    DENIS_ARCHIVE_EXPECTED_SHA256,
    DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
    DENIS_SOURCE_ID,
    DenisArchiveAuditError,
    DenisRecord,
    inspect_denis_archive,
)
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.denis_selection import (
    DENIS_TARGET_PAIRS,
    DenisPreQaSelection,
    DenisPreQaSelectionEntry,
    select_denis_pre_qa_candidate,
)
from kds.eval.denis_source_screen import screen_denis_source_records

SELECTION_FIELDS = (
    "selection_rank",
    "sample_id",
    "member_stem",
    "category",
    "parent_group_id",
    "speaker_pseudo_id",
    "text_id",
    "literal_text_sha256",
    "whitespace_canonical_text_sha256",
    "nfkc_whitespace_canonical_text_sha256",
    "source_audio_sha256",
    "source_audio_size_bytes",
)


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateExposureError(f"Cannot read {label}: {path}.") from error
    if not isinstance(payload, dict):
        raise CandidateExposureError(f"{label} must be a JSON object.")
    return cast(dict[str, object], payload)


def _require_source_audit(path: Path) -> dict[str, object]:
    receipt = _load_json(path, "Denis source audit receipt")
    expected: dict[str, object] = {
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
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise DenisArchiveAuditError(
                f"Denis source audit receipt has unexpected {field!r}: {receipt.get(field)!r}."
            )
    return receipt


def _entry(entry: DenisPreQaSelectionEntry) -> dict[str, object]:
    return {field: getattr(entry, field) for field in SELECTION_FIELDS}


def _write_selection(path: Path, selection: DenisPreQaSelection) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_entry(entry) for entry in selection.entries)


def _require_current_source_screen(
    *,
    path: Path,
    source_audit_path: Path,
    records: tuple[DenisRecord, ...],
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
) -> Mapping[str, object]:
    receipt = _load_json(path, "Denis source-exposure screen")
    created_at = receipt.get("created_at")
    if not isinstance(created_at, str):
        raise CandidateExposureError("Denis source-exposure screen lacks created_at.")
    expected = screen_denis_source_records(
        records=records,
        project_root=project_root,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at=created_at,
        source_audit_receipt={
            "path": str(source_audit_path),
            "sha256": sha256_file(source_audit_path),
        },
    )
    if receipt != expected:
        raise CandidateExposureError(
            "Denis source-exposure screen differs from current pinned project inputs."
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-audit-receipt", type=Path, required=True)
    parser.add_argument("--source-exposure-screen", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--selected-at", required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = (arguments.output_selection, arguments.output_receipt)
    try:
        if (
            len(set(outputs)) != len(outputs)
            or any(path.exists() or not path.parent.is_dir() for path in outputs)
        ):
            raise CandidateExposureError(
                "Denis selection outputs must be distinct, new, and have existing parents."
            )
        project_root = arguments.project_root.resolve(strict=True)
        source_audit = _require_source_audit(arguments.source_audit_receipt)
        inspection = inspect_denis_archive(arguments.archive)
        if (
            inspection.audit.record_identity_fingerprint
            != source_audit.get("record_identity_fingerprint")
        ):
            raise DenisArchiveAuditError("Denis archive differs from its source audit receipt.")
        screen = _require_current_source_screen(
            path=arguments.source_exposure_screen,
            source_audit_path=arguments.source_audit_receipt,
            records=inspection.records,
            project_root=project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
        )
        selection = select_denis_pre_qa_candidate(
            records=inspection.records,
            source_exposure_screen=screen,
            selection_seed=arguments.selection_seed,
            requested_records=DENIS_TARGET_PAIRS,
            target_pairs=DENIS_TARGET_PAIRS,
            selected_at=arguments.selected_at,
        )
        stage = Path(
            tempfile.mkdtemp(prefix=".kds-denis-selection-", dir=arguments.output_receipt.parent)
        )
        try:
            staged_selection = stage / arguments.output_selection.name
            _write_selection(staged_selection, selection)
            receipt = {
                **selection.receipt,
                "archive": {
                    "expected_size_bytes": DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": DENIS_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_selection": True,
                },
                "inputs": {
                    "source_audit_receipt": {
                        "path": str(arguments.source_audit_receipt),
                        "sha256": sha256_file(arguments.source_audit_receipt),
                    },
                    "source_exposure_screen": {
                        "path": str(arguments.source_exposure_screen),
                        "sha256": sha256_file(arguments.source_exposure_screen),
                    },
                },
                "output_selection": {
                    "path": arguments.output_selection.as_posix(),
                    "sha256": sha256_file(staged_selection),
                    "rows": len(selection.entries),
                },
            }
            staged_receipt = stage / arguments.output_receipt.name
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in outputs):
                raise CandidateExposureError("A Denis selection output appeared while staging.")
            staged_selection.replace(arguments.output_selection)
            staged_receipt.replace(arguments.output_receipt)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    except (CandidateExposureError, DenisArchiveAuditError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_records": len(selection.entries),
                "selection": str(arguments.output_selection),
                "selection_sha256": sha256_file(arguments.output_selection),
                "receipt": str(arguments.output_receipt),
                "receipt_sha256": sha256_file(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
