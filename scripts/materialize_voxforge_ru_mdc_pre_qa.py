"""Materialize and technically QA only the frozen VoxForge RU pre-QA selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, validate_manifest, write_manifest
from kds.data.preprocess import preprocess_rows
from kds.data.voxforge import (
    VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
    VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
    VOXFORGE_RU_ARCHIVE_ROOT,
    VOXFORGE_RU_LICENSE,
    VOXFORGE_RU_SOURCE_ID,
    VoxForgeRuAuditError,
    VoxForgeRuRecord,
    load_voxforge_ru_metadata,
)
from kds.eval.voxforge_metadata_screen import voxforge_metadata_identity

_SELECTION_FIELDS = (
    "selection_rank",
    "sample_id",
    "submission_pseudo_id",
    "prompt_id",
    "parent_group_id",
    "speaker_pseudo_id",
    "prompt_text_hash",
    "original_prompt_text_hash",
)
_HEX = frozenset("0123456789abcdef")


class VoxForgePreQaMaterializationError(ValueError):
    """Raised when the frozen selection cannot safely become local QA inputs."""


@dataclass(frozen=True, slots=True)
class FrozenSelectionRow:
    """One privacy-preserving selected metadata row."""

    selection_rank: int
    sample_id: str
    submission_pseudo_id: str
    prompt_id: str
    parent_group_id: str
    speaker_pseudo_id: str
    prompt_text_hash: str
    original_prompt_text_hash: str


@dataclass(frozen=True, slots=True)
class BoundSelectionRow:
    """One frozen selection row bound to one exact source record in memory only."""

    selection: FrozenSelectionRow
    record: VoxForgeRuRecord


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgePreQaMaterializationError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise VoxForgePreQaMaterializationError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise VoxForgePreQaMaterializationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _project_file(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise VoxForgePreQaMaterializationError(f"{label} path must be non-empty.")
    path = (project_root / value).resolve(strict=True)
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise VoxForgePreQaMaterializationError(f"{label} path escapes project root.") from error
    return path


def load_frozen_selection(
    selection_csv: Path, selection_receipt: Path, project_root: Path
) -> tuple[FrozenSelectionRow, ...]:
    """Load only the byte-pinned completed metadata selection."""

    project_root = project_root.resolve(strict=True)
    selection_path = selection_csv.resolve(strict=True)
    receipt_path = selection_receipt.resolve(strict=True)
    try:
        selection_path.relative_to(project_root)
        receipt_path.relative_to(project_root)
    except ValueError as error:
        raise VoxForgePreQaMaterializationError(
            "Frozen selection and receipt must live beneath the project root."
        ) from error
    receipt = _json_object(receipt_path, "VoxForge pre-QA selection receipt")
    output = receipt.get("output_selection")
    policy = receipt.get("selection_policy")
    claims = receipt.get("claims")
    archive = receipt.get("archive")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != "voxforge-ru-mdc-pre-qa-selection-v1"
        or not isinstance(output, Mapping)
        or output.get("path") != selection_csv.as_posix()
        or not isinstance(policy, Mapping)
        or policy.get("selected_records") != 81
        or policy.get("selected_contributor_groups") != 81
        or policy.get("selected_canonical_prompt_text_groups") != 81
        or policy.get("post_selection_backfill") is not False
        or policy.get("selection_uses_audio_or_duration") is not False
        or policy.get("selection_uses_detector_or_model_output") is not False
        or not isinstance(claims, Mapping)
        or claims.get("selection_frozen") is not True
        or claims.get("audio_extraction_performed") is not False
        or claims.get("qa_rejects_must_not_trigger_backfill") is not True
        or not isinstance(archive, Mapping)
        or archive.get("expected_size_bytes") != VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive.get("expected_sha256") != VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256
    ):
        raise VoxForgePreQaMaterializationError("Frozen selection receipt contract is invalid.")
    if sha256_file(selection_path) != _sha256(output.get("sha256"), "Selection CSV SHA-256"):
        raise VoxForgePreQaMaterializationError("Selection CSV differs from its immutable receipt.")
    try:
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != _SELECTION_FIELDS:
                raise VoxForgePreQaMaterializationError("Selection CSV schema is invalid.")
            raw_rows = list(reader)
    except OSError as error:
        raise VoxForgePreQaMaterializationError(f"Cannot read selection CSV: {error}") from error
    if len(raw_rows) != 81:
        raise VoxForgePreQaMaterializationError("Selection CSV must contain exactly 81 rows.")
    selection: list[FrozenSelectionRow] = []
    sample_ids: set[str] = set()
    groups: set[str] = set()
    text_hashes: set[str] = set()
    for expected_rank, row in enumerate(raw_rows, start=1):
        try:
            item = FrozenSelectionRow(
                selection_rank=int(row.get("selection_rank") or ""),
                sample_id=(row.get("sample_id") or "").strip(),
                submission_pseudo_id=(row.get("submission_pseudo_id") or "").strip(),
                prompt_id=(row.get("prompt_id") or "").strip(),
                parent_group_id=(row.get("parent_group_id") or "").strip(),
                speaker_pseudo_id=(row.get("speaker_pseudo_id") or "").strip(),
                prompt_text_hash=_sha256(row.get("prompt_text_hash"), "prompt_text_hash"),
                original_prompt_text_hash=_sha256(
                    row.get("original_prompt_text_hash"), "original_prompt_text_hash"
                ),
            )
        except ValueError as error:
            raise VoxForgePreQaMaterializationError(
                f"Selection row {expected_rank + 1} has an invalid rank."
            ) from error
        if (
            item.selection_rank != expected_rank
            or not item.prompt_id
            or item.sample_id in sample_ids
            or item.parent_group_id in groups
            or item.prompt_text_hash in text_hashes
            or item.parent_group_id != item.speaker_pseudo_id
            or not item.sample_id.startswith(f"{VOXFORGE_RU_SOURCE_ID}:submission:")
            or not item.parent_group_id.startswith(f"{VOXFORGE_RU_SOURCE_ID}:contributor:")
        ):
            raise VoxForgePreQaMaterializationError(
                f"Selection row {expected_rank + 1} violates the frozen selection contract."
            )
        selection.append(item)
        sample_ids.add(item.sample_id)
        groups.add(item.parent_group_id)
        text_hashes.add(item.prompt_text_hash)
    return tuple(selection)


def bind_selection(
    selection: Sequence[FrozenSelectionRow], records: Sequence[VoxForgeRuRecord]
) -> tuple[BoundSelectionRow, ...]:
    """Require every selected pseudonym to match its exact pinned archive metadata."""

    indexed = {voxforge_metadata_identity(record).sample_id: record for record in records}
    if len(indexed) != len(records):
        raise VoxForgePreQaMaterializationError("Pinned VoxForge archive has duplicate sample IDs.")
    bound: list[BoundSelectionRow] = []
    for selection_row in selection:
        record = indexed.get(selection_row.sample_id)
        if record is None:
            raise VoxForgePreQaMaterializationError(
                f"Frozen selection is absent from the pinned archive: {selection_row.sample_id}."
            )
        identity = voxforge_metadata_identity(record)
        if (
            identity.sample_id != selection_row.sample_id
            or hashlib.sha256(record.submission_id.encode("utf-8")).hexdigest()
            != selection_row.submission_pseudo_id
            or record.prompt_id != selection_row.prompt_id
            or identity.parent_group_id != selection_row.parent_group_id
            or identity.speaker_pseudo_id != selection_row.speaker_pseudo_id
            or identity.prompt_text_hash != selection_row.prompt_text_hash
            or identity.original_prompt_text_hash != selection_row.original_prompt_text_hash
        ):
            raise VoxForgePreQaMaterializationError(
                f"Pinned archive binding changed for {selection_row.sample_id}."
            )
        bound.append(BoundSelectionRow(selection_row, record))
    return tuple(bound)


def _safe_member_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise VoxForgePreQaMaterializationError("VoxForge TAR has an unsafe member path.")
    return path


def extract_selected_wavs(
    archive_path: Path, bound: Sequence[BoundSelectionRow], destination: Path
) -> dict[str, Path]:
    """Atomically extract only the selected regular WAV members with opaque filenames."""

    if destination.exists() or not destination.parent.is_dir():
        raise VoxForgePreQaMaterializationError(
            "Raw destination must be new and have an existing parent directory."
        )
    wanted: dict[str, BoundSelectionRow] = {}
    for row in bound:
        member_path = (
            f"{VOXFORGE_RU_ARCHIVE_ROOT}/{row.record.submission_id}/wav/{row.record.prompt_id}.wav"
        )
        if member_path in wanted:
            raise VoxForgePreQaMaterializationError(
                "Frozen selection maps multiple rows to one WAV."
            )
        wanted[member_path] = row
    stage = Path(tempfile.mkdtemp(prefix=".kds-voxforge-pre-qa-", dir=destination.parent))
    staged_destination = stage / destination.name
    staged_destination.mkdir()
    extracted: dict[str, Path] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                _safe_member_path(member.name)
                if not (member.isdir() or member.isfile()):
                    raise VoxForgePreQaMaterializationError(
                        "VoxForge TAR has an unsafe member type."
                    )
                selected = wanted.get(member.name)
                if selected is None:
                    continue
                if not member.isfile() or member.size <= 44:
                    raise VoxForgePreQaMaterializationError(
                        "Selected VoxForge WAV member is invalid."
                    )
                name = f"voxforge_ru_{selected.selection.selection_rank:03d}.wav"
                target = staged_destination / name
                if target.exists():
                    raise VoxForgePreQaMaterializationError("Selected raw destination collided.")
                source = archive.extractfile(member)
                if source is None:
                    raise VoxForgePreQaMaterializationError(
                        "Cannot read selected VoxForge WAV member."
                    )
                with source, target.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
                try:
                    with wave.open(str(target), "rb") as audio:
                        if audio.getnframes() <= 0 or audio.getframerate() != 48_000:
                            raise VoxForgePreQaMaterializationError(
                                "Selected VoxForge WAV does not retain expected 48 kHz audio."
                            )
                except wave.Error as error:
                    raise VoxForgePreQaMaterializationError(
                        "Selected VoxForge WAV is unreadable."
                    ) from error
                extracted[selected.selection.sample_id] = target
        if set(extracted) != {row.selection.sample_id for row in bound}:
            raise VoxForgePreQaMaterializationError("Pinned archive lacks a selected WAV member.")
        staged_destination.replace(destination)
    except (OSError, tarfile.TarError) as error:
        raise VoxForgePreQaMaterializationError(
            f"Cannot safely extract VoxForge WAVs: {error}"
        ) from error
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {sample_id: destination / path.name for sample_id, path in extracted.items()}


def _relative_to_data_root(data_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(data_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise VoxForgePreQaMaterializationError("Output path escapes data root.") from error


def _raw_manifest_rows(
    bound: Sequence[BoundSelectionRow],
    extracted: Mapping[str, Path],
    data_root: Path,
    created_at: str,
) -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    for bound_row in bound:
        source = extracted.get(bound_row.selection.sample_id)
        if source is None:
            raise VoxForgePreQaMaterializationError("Missing selected extraction result.")
        with wave.open(str(source), "rb") as audio:
            duration_s = audio.getnframes() / audio.getframerate()
            sample_rate = audio.getframerate()
        rows.append(
            ManifestRow(
                sample_id=bound_row.selection.sample_id,
                relative_path=_relative_to_data_root(data_root, source),
                sha256=sha256_file(source),
                split="test",
                label="bonafide",
                language="ru",
                code_switch="unknown",
                parent_group_id=bound_row.selection.parent_group_id,
                source_name=VOXFORGE_RU_SOURCE_ID,
                source_license=VOXFORGE_RU_LICENSE,
                rights_basis=(
                    "Pinned VoxForge Russian GPL-3.0-or-later archive; personal research only"
                ),
                speaker_pseudo_id=bound_row.selection.speaker_pseudo_id,
                text_id=bound_row.selection.prompt_id,
                text_hash=bound_row.selection.prompt_text_hash,
                duration_s=duration_s,
                generator_family="",
                generator_name="",
                generator_version="",
                voice_id="",
                clone_consent_id="",
                device="unknown",
                capture_route="voxforge_submission_read_speech",
                original_sr=sample_rate,
                codec="wav",
                augmentation_chain="",
                augmentation_seed="",
                created_at=created_at,
            )
        )
    return tuple(rows)


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
    outputs = (arguments.raw_manifest, arguments.ready_manifest, arguments.materialization_receipt)
    try:
        datetime.fromisoformat(arguments.materialized_at.replace("Z", "+00:00"))
        if len(set(outputs)) != len(outputs) or any(
            path.exists() or not path.parent.is_dir() for path in outputs
        ):
            raise VoxForgePreQaMaterializationError("Manifest/receipt outputs must be new.")
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        _relative_to_data_root(data_root, arguments.raw_destination.parent)
        selection = load_frozen_selection(
            arguments.selection_csv, arguments.selection_receipt, project_root
        )
        records = load_voxforge_ru_metadata(arguments.archive)
        bound = bind_selection(selection, records)
        extracted = extract_selected_wavs(arguments.archive, bound, arguments.raw_destination)
        raw_rows = _raw_manifest_rows(bound, extracted, data_root, arguments.materialized_at)
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
            prefix="kds-voxforge-pre-qa-receipt-", dir=arguments.materialization_receipt.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged_raw = stage / arguments.raw_manifest.name
            staged_ready = stage / arguments.ready_manifest.name
            staged_receipt = stage / arguments.materialization_receipt.name
            write_manifest(staged_raw, raw_rows)
            write_manifest(staged_ready, prepared.processed_rows)
            receipt = {
                "schema_version": 1,
                "protocol_id": "voxforge-ru-mdc-pre-qa-materialization-v1",
                "materialized_at": arguments.materialized_at,
                "candidate_state": "frozen bona-fide extraction and technical decode/QA/VAD only",
                "archive": {
                    "expected_size_bytes": VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
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
                    "one_record_per_prompt_text_and_contributor_group": True,
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
                    "replacement_or_backfill": False,
                },
                "claims": {
                    "audio_extraction_performed": True,
                    "synthetic_audio_generated": False,
                    "technical_decode_qa_vad_performed": True,
                    "acoustic_review_performed": False,
                    "pairing_performed": False,
                    "detector_inference_performed": False,
                    "detector_inference_authorized": False,
                    "future_synthesis_must_use_only_ready_frozen_texts": True,
                },
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staged_raw.replace(arguments.raw_manifest)
            staged_ready.replace(arguments.ready_manifest)
            staged_receipt.replace(arguments.materialization_receipt)
    except (
        LicenseLedgerError,
        ManifestError,
        VoxForgePreQaMaterializationError,
        VoxForgeRuAuditError,
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
                "receipt": str(arguments.materialization_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
