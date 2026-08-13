"""Extract and QA/VAD only the frozen Common Voice RU / Silero V5.5 pre-QA selection."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import require_valid_assets, sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
    COMMON_VOICE_RU_V24_SOURCE_ID,
    CommonVoiceIngestionError,
    CommonVoiceRecord,
    ExtractedCommonVoiceAsset,
    common_voice_manifest_rows,
    extract_common_voice_audio_slice,
    inspect_extracted_common_voice_audio,
    load_common_voice_metadata_from_archive,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, validate_manifest, write_manifest
from kds.data.preprocess import preprocess_rows
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.common_voice_metadata_screen import common_voice_metadata_identity

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
_HEX = frozenset("0123456789abcdef")


class CommonVoicePreQaMaterializationError(ValueError):
    """Raised when the immutable selection cannot safely become local QA inputs."""


@dataclass(frozen=True, slots=True)
class FrozenPreQaSelectionRow:
    selection_rank: int
    sample_id: str
    clip_name: str
    source_split: str
    parent_group_id: str
    speaker_pseudo_id: str
    text_id: str
    text_hash: str


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CommonVoicePreQaMaterializationError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise CommonVoicePreQaMaterializationError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise CommonVoicePreQaMaterializationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _project_file(project_root: Path, relative_path: object, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise CommonVoicePreQaMaterializationError(f"{label} path must be non-empty.")
    candidate = (project_root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise CommonVoicePreQaMaterializationError(f"{label} escapes the project root.") from error
    return candidate


def _require_receipt_file_input(
    project_root: Path, inputs: Mapping[str, object], name: str
) -> None:
    value = inputs.get(name)
    if not isinstance(value, Mapping):
        raise CommonVoicePreQaMaterializationError(f"Selection receipt has no {name} binding.")
    path = _project_file(project_root, value.get("path"), f"Selection receipt {name}")
    if sha256_file(path) != _sha256(value.get("sha256"), f"Selection receipt {name} SHA-256"):
        raise CommonVoicePreQaMaterializationError(
            f"Selection receipt {name} bytes no longer match their recorded SHA-256."
        )


def load_frozen_pre_qa_selection(
    selection_csv: Path, selection_receipt: Path, project_root: Path
) -> tuple[FrozenPreQaSelectionRow, ...]:
    """Load only a CSV pinned by the completed immutable selection receipt."""

    project_root = project_root.resolve(strict=True)
    selection_path = selection_csv.resolve(strict=True)
    receipt_path = selection_receipt.resolve(strict=True)
    try:
        selection_path.relative_to(project_root)
        receipt_path.relative_to(project_root)
    except ValueError as error:
        raise CommonVoicePreQaMaterializationError(
            "Frozen selection and receipt must live under the project root."
        ) from error
    receipt = _json_object(receipt_path, "pre-QA selection receipt")
    output = receipt.get("output_selection")
    policy = receipt.get("selection_policy")
    claims = receipt.get("claims")
    inputs = receipt.get("inputs")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id")
        != "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-selection-v1"
        or not isinstance(output, Mapping)
        or output.get("path") != selection_csv.as_posix()
        or not isinstance(policy, Mapping)
        or policy.get("kind") != "seeded_two_stage_one_record_per_client_group"
        or policy.get("post_selection_backfill") is not False
        or policy.get("selection_uses_audio_or_duration") is not False
        or policy.get("selection_uses_detector_or_model_output") is not False
        or policy.get("selection_uses_model_metrics_or_final_errors") is not False
        or not isinstance(claims, Mapping)
        or claims.get("selection_frozen") is not True
        or claims.get("audio_extraction_performed") is not False
        or claims.get("future_extraction_must_use_only_selected_clip_names") is not True
        or claims.get("qa_rejects_must_not_trigger_backfill") is not True
        or not isinstance(inputs, Mapping)
    ):
        raise CommonVoicePreQaMaterializationError(
            "Pre-QA selection receipt has an invalid immutable-selection contract."
        )
    if sha256_file(selection_path) != _sha256(output.get("sha256"), "Selection CSV SHA-256"):
        raise CommonVoicePreQaMaterializationError(
            "Pre-QA selection CSV bytes do not match its write-once receipt."
        )
    for name in (
        "metadata_exposure_screen",
        "literal_text_screen",
        "silero_v5_5_model_lock",
    ):
        _require_receipt_file_input(project_root, inputs, name)

    try:
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != SELECTION_FIELDS:
                raise CommonVoicePreQaMaterializationError(
                    "Pre-QA selection CSV has an invalid schema."
                )
            source_rows = list(reader)
    except OSError as error:
        raise CommonVoicePreQaMaterializationError(
            f"Cannot read pre-QA selection CSV: {error}"
        ) from error
    selected_records = policy.get("selected_records")
    selected_groups = policy.get("selected_client_groups")
    if (
        not isinstance(selected_records, int)
        or not isinstance(selected_groups, int)
        or output.get("rows") != selected_records
        or len(source_rows) != selected_records
    ):
        raise CommonVoicePreQaMaterializationError(
            "Pre-QA selection receipt does not agree with CSV row count."
        )

    selection: list[FrozenPreQaSelectionRow] = []
    sample_ids: set[str] = set()
    groups: set[str] = set()
    text_hashes: set[str] = set()
    for expected_rank, row in enumerate(source_rows, start=1):
        try:
            rank = int(row.get("selection_rank") or "")
        except ValueError as error:
            raise CommonVoicePreQaMaterializationError(
                f"Pre-QA selection row {expected_rank + 1} has an invalid selection_rank."
            ) from error
        item = FrozenPreQaSelectionRow(
            selection_rank=rank,
            sample_id=(row.get("sample_id") or "").strip(),
            clip_name=(row.get("clip_name") or "").strip(),
            source_split=(row.get("source_split") or "").strip(),
            parent_group_id=(row.get("parent_group_id") or "").strip(),
            speaker_pseudo_id=(row.get("speaker_pseudo_id") or "").strip(),
            text_id=(row.get("text_id") or "").strip(),
            text_hash=(row.get("text_hash") or "").strip(),
        )
        expected_sample = f"{COMMON_VOICE_RU_V24_SOURCE_ID}:{Path(item.clip_name).stem}"
        if (
            item.selection_rank != expected_rank
            or item.sample_id != expected_sample
            or item.source_split != "test"
            or not item.clip_name.endswith(".mp3")
            or not item.text_id
            or _sha256(item.text_hash, f"Pre-QA selection row {expected_rank + 1} text_hash")
            != item.text_hash
            or item.parent_group_id != item.speaker_pseudo_id
            or not item.parent_group_id.startswith(f"{COMMON_VOICE_RU_V24_SOURCE_ID}:client:")
            or item.sample_id in sample_ids
            or item.parent_group_id in groups
            or item.text_hash in text_hashes
        ):
            raise CommonVoicePreQaMaterializationError(
                f"Pre-QA selection row {expected_rank + 1} violates the frozen selection contract."
            )
        selection.append(item)
        sample_ids.add(item.sample_id)
        groups.add(item.parent_group_id)
        text_hashes.add(item.text_hash)
    if len(groups) != selected_groups:
        raise CommonVoicePreQaMaterializationError(
            "Pre-QA selection receipt client-group count does not match the CSV."
        )
    return tuple(selection)


def bind_frozen_pre_qa_selection(
    selection: Sequence[FrozenPreQaSelectionRow], records: Sequence[CommonVoiceRecord]
) -> tuple[CommonVoiceRecord, ...]:
    """Bind every frozen CSV identity to the pinned archive's exact test TSV metadata."""

    records_by_sample = {
        common_voice_metadata_identity(record).sample_id: record for record in records
    }
    if len(records_by_sample) != len(records):
        raise CommonVoicePreQaMaterializationError(
            "Common Voice archive test metadata has duplicate sample IDs."
        )
    bound: list[CommonVoiceRecord] = []
    for item in selection:
        record = records_by_sample.get(item.sample_id)
        if record is None:
            raise CommonVoicePreQaMaterializationError(
                f"Frozen pre-QA sample is absent from the pinned archive: {item.sample_id}."
            )
        identity = common_voice_metadata_identity(record)
        if (
            record.clip_name != item.clip_name
            or record.split != item.source_split
            or identity.parent_group_id != item.parent_group_id
            or identity.speaker_pseudo_id != item.speaker_pseudo_id
            or record.sentence_id != item.text_id
            or identity.text_hash != item.text_hash
        ):
            raise CommonVoicePreQaMaterializationError(
                f"Pinned archive metadata differs from frozen pre-QA row {item.sample_id}."
            )
        bound.append(record)
    return tuple(bound)


def _relative_to_data_root(data_root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise CommonVoicePreQaMaterializationError(f"{label} must live below data root.") from error


def _validate_outputs(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths) or any(
        path.exists() or not path.parent.is_dir() for path in paths
    ):
        raise CommonVoicePreQaMaterializationError(
            "Materialization outputs must be distinct, new, and have existing parents."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--raw-destination", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--materialized-at", required=True)
    arguments = parser.parse_args()
    outputs = (
        arguments.raw_manifest,
        arguments.ready_manifest,
        arguments.materialization_receipt,
    )
    try:
        datetime.fromisoformat(arguments.materialized_at.replace("Z", "+00:00"))
        _validate_outputs(outputs)
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        _relative_to_data_root(data_root, arguments.raw_destination, "Raw destination")
        if arguments.raw_destination.exists() or not arguments.raw_destination.parent.is_dir():
            raise CommonVoicePreQaMaterializationError(
                "Raw extraction destination must be new and have an existing parent."
            )
        selection = load_frozen_pre_qa_selection(
            arguments.selection_csv, arguments.selection_receipt, project_root
        )
        records = load_common_voice_metadata_from_archive(arguments.archive, ("test",))
        bound_records = bind_frozen_pre_qa_selection(selection, records)
        extracted = extract_common_voice_audio_slice(
            arguments.archive,
            (record.clip_name for record in bound_records),
            arguments.raw_destination,
        )
        assets: dict[str, ExtractedCommonVoiceAsset] = {}
        for record in bound_records:
            extracted_path = extracted.get(record.clip_name)
            if extracted_path is None:
                raise CommonVoicePreQaMaterializationError(
                    f"Frozen clip was not extracted: {record.clip_name}."
                )
            duration_s, original_sr = inspect_extracted_common_voice_audio(extracted_path)
            assets[record.clip_name] = ExtractedCommonVoiceAsset(
                clip_name=record.clip_name,
                relative_path=_relative_to_data_root(data_root, extracted_path, "Extracted clip"),
                sha256=sha256_file(extracted_path),
                duration_s=duration_s,
                original_sr=original_sr,
            )
        raw_rows = common_voice_manifest_rows(
            bound_records, assets, created_at=arguments.materialized_at
        )
        validate_manifest(raw_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(raw_rows, ledger)
        require_valid_assets(raw_rows, data_root)
        prepared = preprocess_rows(
            raw_rows, data_root, AudioPreparationPipeline(), allow_rejections=True
        )
        validate_manifest(prepared.processed_rows)
        validate_manifest_licenses(prepared.processed_rows, ledger)
        require_valid_assets(prepared.processed_rows, data_root)
        with tempfile.TemporaryDirectory(
            prefix="kds-common-voice-pre-qa-", dir=arguments.materialization_receipt.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged_raw = stage / arguments.raw_manifest.name
            staged_ready = stage / arguments.ready_manifest.name
            staged_receipt = stage / arguments.materialization_receipt.name
            write_manifest(staged_raw, raw_rows)
            write_manifest(staged_ready, prepared.processed_rows)
            receipt = {
                "schema_version": 1,
                "protocol_id": "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-materialization-v1",
                "materialized_at": arguments.materialized_at,
                "candidate_state": (
                    "frozen bona-fide extraction and normal decode/QA/VAD only; no synthetic "
                    "audio, acoustic review, pairing, or detector inference"
                ),
                "archive": {
                    "path": str(arguments.archive),
                    "expected_size_bytes": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_metadata_read_and_extraction": True,
                },
                "selection": {
                    "csv": {
                        "path": arguments.selection_csv.as_posix(),
                        "sha256": sha256_file(arguments.selection_csv),
                        "rows": len(selection),
                    },
                    "receipt": {
                        "path": arguments.selection_receipt.as_posix(),
                        "sha256": sha256_file(arguments.selection_receipt),
                    },
                    "one_record_per_client_group": True,
                    "post_selection_backfill": False,
                },
                "outputs": {
                    "raw_manifest": {
                        "path": arguments.raw_manifest.as_posix(),
                        "sha256": sha256_file(staged_raw),
                        "rows": len(raw_rows),
                    },
                    "ready_manifest": {
                        "path": arguments.ready_manifest.as_posix(),
                        "sha256": sha256_file(staged_ready),
                        "rows": len(prepared.processed_rows),
                    },
                },
                "technical_qa": {
                    "pipeline": (
                        "AudioPreparationPipeline: decode, mono PCM WAV 16 kHz, quality checks, "
                        "WebRTC VAD"
                    ),
                    "raw_rows": len(raw_rows),
                    "ready_rows": len(prepared.processed_rows),
                    "rejected_rows": [
                        {
                            "sample_id": issue.sample_id,
                            "relative_path": issue.relative_path,
                            "detail": issue.detail,
                        }
                        for issue in prepared.issues
                    ],
                    "reused_rows": 0,
                    "replacement_or_backfill": False,
                },
                "claims": {
                    "audio_extraction_performed": True,
                    "synthetic_audio_generated": False,
                    "technical_decode_qa_vad_performed": True,
                    "acoustic_review_performed": False,
                    "detector_inference_performed": False,
                    "detector_inference_authorized": False,
                    "future_synthesis_must_use_only_ready_frozen_texts": True,
                },
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in outputs):
                raise CommonVoicePreQaMaterializationError(
                    "A materialization output appeared while staging."
                )
            staged_raw.replace(arguments.raw_manifest)
            staged_ready.replace(arguments.ready_manifest)
            staged_receipt.replace(arguments.materialization_receipt)
    except (
        CandidateExposureError,
        CommonVoiceIngestionError,
        CommonVoicePreQaMaterializationError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        ValueError,
    ) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (LicenseLedgerError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "raw_rows": len(raw_rows),
                "ready_rows": len(prepared.processed_rows),
                "rejected_rows": len(prepared.issues),
                "raw_manifest": str(arguments.raw_manifest),
                "ready_manifest": str(arguments.ready_manifest),
                "receipt": str(arguments.materialization_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
