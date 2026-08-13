"""Freeze a one-record-per-group Common Voice RU V5.5 pre-QA candidate before extraction."""

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
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
    CommonVoiceIngestionError,
    CommonVoiceRecord,
    load_common_voice_metadata_from_archive,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.data.silero_v5_5 import load_silero_v5_5_runtime
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.common_voice_metadata_screen import (
    CommonVoiceMetadataScreen,
    CommonVoicePreQaSelection,
    CommonVoicePreQaSelectionEntry,
    CommonVoiceTextCompatibilityScreen,
    screen_common_voice_ru_test_metadata,
    screen_silero_v5_5_literal_text_compatibility,
    select_common_voice_ru_v24_silero_v5_5_pre_qa_candidate,
)

SELECTION_FIELDS = (
    "selection_rank",
    "sample_id",
    "clip_name",
    "source_split",
    "parent_group_id",
    "speaker_pseudo_id",
    "text_id",
    "text_hash",
)


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateExposureError(f"Cannot read {label}: {error}") from error
    if not isinstance(raw, dict):
        raise CandidateExposureError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], raw)


def _entry_mapping(entry: CommonVoicePreQaSelectionEntry) -> dict[str, object]:
    return {
        "selection_rank": entry.selection_rank,
        "sample_id": entry.sample_id,
        "clip_name": entry.clip_name,
        "source_split": entry.source_split,
        "parent_group_id": entry.parent_group_id,
        "speaker_pseudo_id": entry.speaker_pseudo_id,
        "text_id": entry.text_id,
        "text_hash": entry.text_hash,
    }


def _write_selection_csv(path: Path, selection: CommonVoicePreQaSelection) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_entry_mapping(entry) for entry in selection.entries)


def _expected_metadata_screen(
    *,
    archive: Path,
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
    created_at: str,
) -> tuple[dict[str, object], CommonVoiceMetadataScreen, tuple[CommonVoiceRecord, ...]]:
    records = tuple(load_common_voice_metadata_from_archive(archive, ("test",)))
    metadata_screen = screen_common_voice_ru_test_metadata(
        records=records,
        project_root=project_root,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at=created_at,
    )
    payload = {
        **metadata_screen.receipt,
        "archive": {
            "path": str(archive),
            "expected_size_bytes": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
            "expected_sha256": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
            "identity_verified_before_metadata_read": True,
        },
    }
    return payload, metadata_screen, records


def _require_exact_parent_screens(
    *,
    archive: Path,
    project_root: Path,
    config_root: Path,
    manifest_root: Path,
    metadata_screen_path: Path,
    literal_text_screen_path: Path,
    model_lock_path: Path,
) -> CommonVoiceTextCompatibilityScreen:
    metadata_receipt = _load_json_object(metadata_screen_path, "metadata screen receipt")
    literal_receipt = _load_json_object(literal_text_screen_path, "literal-text screen receipt")
    metadata_created_at = metadata_receipt.get("created_at")
    literal_audited_at = literal_receipt.get("audited_at")
    if (
        not isinstance(metadata_created_at, str)
        or not isinstance(literal_audited_at, str)
        or metadata_created_at != literal_audited_at
    ):
        raise CandidateExposureError(
            "Parent metadata and literal-text screens must share one ISO-8601 audit timestamp."
        )
    expected_metadata, metadata_screen, records = _expected_metadata_screen(
        archive=archive,
        project_root=project_root,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at=metadata_created_at,
    )
    if metadata_receipt != expected_metadata:
        raise CandidateExposureError(
            "Metadata screen receipt does not match the current pinned archive/config/manifest "
            "inputs. Publish a new screen before selecting."
        )
    lock = load_research_tts_model_lock(model_lock_path)
    if len(lock.models) != 1:
        raise ResearchTtsError("Pre-QA selection requires exactly one V5.5 model lock entry.")
    model = lock.models[0]
    runtime = load_silero_v5_5_runtime(model)
    compatibility = screen_silero_v5_5_literal_text_compatibility(
        records=records,
        metadata_screen=metadata_screen,
    )
    input_metadata_screen = compatibility.receipt["input_metadata_screen"]
    if not isinstance(input_metadata_screen, dict):
        raise CandidateExposureError("Literal-text screen has an invalid input-screen receipt.")
    expected_literal = {
        **compatibility.receipt,
        "audited_at": literal_audited_at,
        "archive": expected_metadata["archive"],
        "input_metadata_screen": {
            **input_metadata_screen,
            "path": str(metadata_screen_path),
            "sha256": sha256_file(metadata_screen_path),
        },
        "model_lock": {
            "path": str(model_lock_path),
            "sha256": sha256_file(model_lock_path),
            "model_id": model.model_id,
            "runtime_kind": model.runtime["kind"],
            "fixed_speaker": runtime.fixed_speaker,
            "sample_rate": runtime.sample_rate,
        },
    }
    if literal_receipt != expected_literal:
        raise CandidateExposureError(
            "Literal-text screen receipt does not match the current pinned parents/model lock. "
            "Publish a new screen before selecting."
        )
    return compatibility


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--metadata-screen", type=Path, required=True)
    parser.add_argument("--literal-text-screen", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--requested-client-groups", type=int, required=True)
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
                "Pre-QA selection outputs must be distinct, new, and have existing parents."
            )
        project_root = arguments.project_root.resolve(strict=True)
        compatibility = _require_exact_parent_screens(
            archive=arguments.archive,
            project_root=project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
            metadata_screen_path=arguments.metadata_screen,
            literal_text_screen_path=arguments.literal_text_screen,
            model_lock_path=arguments.model_lock,
        )
        selection = select_common_voice_ru_v24_silero_v5_5_pre_qa_candidate(
            compatibility_screen=compatibility,
            selection_seed=arguments.selection_seed,
            requested_client_groups=arguments.requested_client_groups,
            selected_at=arguments.selected_at,
        )
        stage = Path(
            tempfile.mkdtemp(prefix=".kds-v5-5-pre-qa-", dir=arguments.output_receipt.parent)
        )
        try:
            staged_selection = stage / arguments.output_selection.name
            _write_selection_csv(staged_selection, selection)
            receipt = {
                **selection.receipt,
                "archive": {
                    "path": str(arguments.archive),
                    "expected_size_bytes": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_metadata_read": True,
                },
                "inputs": {
                    "metadata_exposure_screen": {
                        "path": str(arguments.metadata_screen),
                        "sha256": sha256_file(arguments.metadata_screen),
                    },
                    "literal_text_screen": {
                        "path": str(arguments.literal_text_screen),
                        "sha256": sha256_file(arguments.literal_text_screen),
                    },
                    "silero_v5_5_model_lock": {
                        "path": str(arguments.model_lock),
                        "sha256": sha256_file(arguments.model_lock),
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
                raise CandidateExposureError("A pre-QA selection output appeared while staging.")
            staged_selection.replace(arguments.output_selection)
            staged_receipt.replace(arguments.output_receipt)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    except (CandidateExposureError, CommonVoiceIngestionError, OSError, ResearchTtsError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "selection": str(arguments.output_selection),
                "receipt": str(arguments.output_receipt),
                "selected_records": len(selection.entries),
                "selected_client_groups": len(
                    {entry.parent_group_id for entry in selection.entries}
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
