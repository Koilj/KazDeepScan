"""Freeze a metadata-only VoxForge RU pre-QA selection before WAV extraction."""

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
from kds.data.voxforge import (
    VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
    VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
    VOXFORGE_RU_SOURCE_ID,
    VoxForgeRuAuditError,
    VoxForgeRuRecord,
    load_voxforge_ru_metadata,
)
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.voxforge_metadata_screen import (
    VoxForgeMetadataScreen,
    VoxForgePreQaSelection,
    VoxForgePreQaSelectionEntry,
    screen_voxforge_ru_metadata,
    select_voxforge_ru_mdc_pre_qa_candidate,
)

SELECTION_FIELDS = (
    "selection_rank",
    "sample_id",
    "submission_pseudo_id",
    "prompt_id",
    "parent_group_id",
    "speaker_pseudo_id",
    "prompt_text_hash",
    "original_prompt_text_hash",
)


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateExposureError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CandidateExposureError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_source_audit(path: Path) -> None:
    receipt = _load_json_object(path, "VoxForge source audit receipt")
    expected = {
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
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise VoxForgeRuAuditError(
                f"VoxForge source audit receipt has unexpected {field!r}: {receipt.get(field)!r}."
            )


def _entry_mapping(entry: VoxForgePreQaSelectionEntry) -> dict[str, object]:
    return {
        "selection_rank": entry.selection_rank,
        "sample_id": entry.sample_id,
        "submission_pseudo_id": entry.submission_pseudo_id,
        "prompt_id": entry.prompt_id,
        "parent_group_id": entry.parent_group_id,
        "speaker_pseudo_id": entry.speaker_pseudo_id,
        "prompt_text_hash": entry.prompt_text_hash,
        "original_prompt_text_hash": entry.original_prompt_text_hash,
    }


def _write_selection_csv(path: Path, selection: VoxForgePreQaSelection) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_entry_mapping(entry) for entry in selection.entries)


def _require_exact_metadata_screen(
    *,
    archive: Path,
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
    source_audit_receipt: Path,
    metadata_screen_path: Path,
) -> tuple[VoxForgeMetadataScreen, tuple[VoxForgeRuRecord, ...]]:
    receipt = _load_json_object(metadata_screen_path, "VoxForge metadata screen receipt")
    created_at = receipt.get("created_at")
    if not isinstance(created_at, str):
        raise CandidateExposureError("VoxForge metadata screen receipt lacks created_at.")
    _require_source_audit(source_audit_receipt)
    records = load_voxforge_ru_metadata(archive)
    screen = screen_voxforge_ru_metadata(
        records=records,
        project_root=project_root,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at=created_at,
    )
    expected = {
        **screen.receipt,
        "archive": {
            "source_audit_receipt": {
                "path": str(source_audit_receipt),
                "sha256": sha256_file(source_audit_receipt),
            },
            "expected_size_bytes": VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
            "expected_sha256": VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
            "identity_verified_before_metadata_read": True,
        },
    }
    if receipt != expected:
        raise CandidateExposureError(
            "VoxForge metadata screen does not match the current pinned archive/config/manifest "
            "inputs. Publish a new screen before selecting."
        )
    return screen, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-audit-receipt", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--metadata-screen", type=Path, required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--requested-text-groups", type=int, required=True)
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
                "VoxForge pre-QA selection outputs must be distinct, new, "
                "and have existing parents."
            )
        project_root = arguments.project_root.resolve(strict=True)
        metadata_screen, records = _require_exact_metadata_screen(
            archive=arguments.archive,
            project_root=project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
            source_audit_receipt=arguments.source_audit_receipt,
            metadata_screen_path=arguments.metadata_screen,
        )
        selection = select_voxforge_ru_mdc_pre_qa_candidate(
            records=records,
            metadata_screen=metadata_screen,
            selection_seed=arguments.selection_seed,
            requested_text_groups=arguments.requested_text_groups,
            selected_at=arguments.selected_at,
        )
        stage = Path(
            tempfile.mkdtemp(prefix=".kds-voxforge-pre-qa-", dir=arguments.output_receipt.parent)
        )
        try:
            staged_selection = stage / arguments.output_selection.name
            _write_selection_csv(staged_selection, selection)
            receipt = {
                **selection.receipt,
                "archive": {
                    "expected_size_bytes": VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_metadata_read": True,
                },
                "inputs": {
                    "source_audit_receipt": {
                        "path": str(arguments.source_audit_receipt),
                        "sha256": sha256_file(arguments.source_audit_receipt),
                    },
                    "metadata_exposure_screen": {
                        "path": str(arguments.metadata_screen),
                        "sha256": sha256_file(arguments.metadata_screen),
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
                raise CandidateExposureError(
                    "A VoxForge pre-QA selection output appeared while staging."
                )
            staged_selection.replace(arguments.output_selection)
            staged_receipt.replace(arguments.output_receipt)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    except (CandidateExposureError, VoxForgeRuAuditError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "selection": str(arguments.output_selection),
                "receipt": str(arguments.output_receipt),
                "selected_records": len(selection.entries),
                "selected_contributor_groups": len(
                    {entry.parent_group_id for entry in selection.entries}
                ),
                "selected_prompt_text_groups": len(
                    {entry.prompt_text_hash for entry in selection.entries}
                ),
                "receipt_sha256": sha256_file(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
