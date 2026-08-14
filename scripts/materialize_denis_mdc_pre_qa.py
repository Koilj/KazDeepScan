"""Extract and technically QA only the frozen 79-row Denis bona-fide selection."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import require_valid_assets, sha256_file
from kds.data.denis import (
    DENIS_ARCHIVE_EXPECTED_SHA256,
    DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
    DENIS_SOURCE_ID,
    DENIS_SOURCE_LICENSE,
    DenisArchiveAuditError,
    DenisRecord,
    inspect_denis_archive,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, validate_manifest, write_manifest
from kds.data.preprocess import PreprocessIssue, preprocess_rows
from kds.eval.denis_selection import DENIS_SINGLE_SPEAKER_GROUP, DENIS_TARGET_PAIRS

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
_HEX = frozenset("0123456789abcdef")


class DenisPreQaMaterializationError(ValueError):
    """Raised when the frozen Denis selection cannot be materialized exactly once."""


@dataclass(frozen=True, slots=True)
class FrozenDenisSelectionRow:
    selection_rank: int
    sample_id: str
    member_stem: str
    category: str
    parent_group_id: str
    speaker_pseudo_id: str
    text_id: str
    literal_text_sha256: str
    whitespace_canonical_text_sha256: str
    nfkc_whitespace_canonical_text_sha256: str
    source_audio_sha256: str
    source_audio_size_bytes: int


@dataclass(frozen=True, slots=True)
class BoundDenisSelectionRow:
    selection: FrozenDenisSelectionRow
    record: DenisRecord


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DenisPreQaMaterializationError(f"Cannot read {label}: {path}.") from error
    if not isinstance(payload, dict):
        raise DenisPreQaMaterializationError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise DenisPreQaMaterializationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _project_file(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DenisPreQaMaterializationError(f"{label} path must be non-empty.")
    path = (project_root / value).resolve(strict=True)
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise DenisPreQaMaterializationError(f"{label} path escapes project root.") from error
    return path


def _require_pinned_input(
    project_root: Path, inputs: Mapping[str, object], name: str
) -> None:
    binding = inputs.get(name)
    if not isinstance(binding, Mapping):
        raise DenisPreQaMaterializationError(f"Selection receipt lacks {name!r} binding.")
    path = _project_file(project_root, binding.get("path"), name)
    if sha256_file(path) != _sha256(binding.get("sha256"), f"{name} SHA-256"):
        raise DenisPreQaMaterializationError(f"Pinned selection input changed: {name}.")


def load_frozen_denis_selection(
    selection_csv: Path, selection_receipt: Path, project_root: Path
) -> tuple[FrozenDenisSelectionRow, ...]:
    """Load the exact write-once 79-row selection and all pinned parent receipts."""

    project_root = project_root.resolve(strict=True)
    selection_path = selection_csv.resolve(strict=True)
    receipt_path = selection_receipt.resolve(strict=True)
    try:
        selection_path.relative_to(project_root)
        receipt_path.relative_to(project_root)
    except ValueError as error:
        raise DenisPreQaMaterializationError(
            "Denis selection and receipt must live beneath the project root."
        ) from error
    receipt = _json_object(receipt_path, "Denis pre-QA selection receipt")
    output = receipt.get("output_selection")
    policy = receipt.get("selection_policy")
    claims = receipt.get("claims")
    archive = receipt.get("archive")
    inputs = receipt.get("inputs")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != "denis-1-0-mdc-pre-qa-selection-v1"
        or not isinstance(output, Mapping)
        or output.get("path") != selection_csv.as_posix()
        or output.get("rows") != DENIS_TARGET_PAIRS
        or not isinstance(policy, Mapping)
        or policy.get("target_pairs") != DENIS_TARGET_PAIRS
        or policy.get("requested_records") != DENIS_TARGET_PAIRS
        or policy.get("selected_records") != DENIS_TARGET_PAIRS
        or policy.get("selected_speaker_groups") != 1
        or policy.get("literal_and_canonical_text_hashes_bound_before_materialization")
        is not True
        or policy.get("single_source_speaker_group_retained_for_every_row") is not True
        or policy.get("post_selection_replacement_or_backfill") is not False
        or policy.get("selection_uses_audio_or_duration") is not False
        or policy.get("selection_uses_audio_quality_or_vad") is not False
        or policy.get("selection_uses_detector_or_model_output") is not False
        or not isinstance(claims, Mapping)
        or claims.get("selection_frozen") is not True
        or claims.get("audio_extraction_performed") is not False
        or claims.get("technical_decode_qa_vad_performed") is not False
        or claims.get("qa_rejects_must_not_trigger_replacement_or_backfill") is not True
        or claims.get("training_data_overlap_unverified") is not True
        or claims.get("single_speaker") is not True
        or claims.get("speaker_independent") is not False
        or not isinstance(archive, Mapping)
        or archive.get("expected_size_bytes") != DENIS_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive.get("expected_sha256") != DENIS_ARCHIVE_EXPECTED_SHA256
        or not isinstance(inputs, Mapping)
    ):
        raise DenisPreQaMaterializationError("Frozen Denis selection receipt is invalid.")
    if sha256_file(selection_path) != _sha256(output.get("sha256"), "Selection CSV SHA-256"):
        raise DenisPreQaMaterializationError("Denis selection CSV differs from its receipt.")
    _require_pinned_input(project_root, inputs, "source_audit_receipt")
    _require_pinned_input(project_root, inputs, "source_exposure_screen")

    try:
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != SELECTION_FIELDS:
                raise DenisPreQaMaterializationError("Denis selection CSV schema is invalid.")
            mappings = list(reader)
    except OSError as error:
        raise DenisPreQaMaterializationError("Cannot read Denis selection CSV.") from error
    if len(mappings) != DENIS_TARGET_PAIRS:
        raise DenisPreQaMaterializationError("Denis selection must contain exactly 79 rows.")

    rows: list[FrozenDenisSelectionRow] = []
    sample_ids: set[str] = set()
    member_stems: set[str] = set()
    literal_hashes: set[str] = set()
    canonical_hashes: set[str] = set()
    audio_hashes: set[str] = set()
    for expected_rank, mapping in enumerate(mappings, start=1):
        try:
            row = FrozenDenisSelectionRow(
                selection_rank=int(mapping.get("selection_rank") or ""),
                sample_id=(mapping.get("sample_id") or "").strip(),
                member_stem=(mapping.get("member_stem") or "").strip(),
                category=(mapping.get("category") or "").strip(),
                parent_group_id=(mapping.get("parent_group_id") or "").strip(),
                speaker_pseudo_id=(mapping.get("speaker_pseudo_id") or "").strip(),
                text_id=(mapping.get("text_id") or "").strip(),
                literal_text_sha256=_sha256(
                    mapping.get("literal_text_sha256"), "literal_text_sha256"
                ),
                whitespace_canonical_text_sha256=_sha256(
                    mapping.get("whitespace_canonical_text_sha256"),
                    "whitespace_canonical_text_sha256",
                ),
                nfkc_whitespace_canonical_text_sha256=_sha256(
                    mapping.get("nfkc_whitespace_canonical_text_sha256"),
                    "nfkc_whitespace_canonical_text_sha256",
                ),
                source_audio_sha256=_sha256(
                    mapping.get("source_audio_sha256"), "source_audio_sha256"
                ),
                source_audio_size_bytes=int(mapping.get("source_audio_size_bytes") or ""),
            )
        except ValueError as error:
            raise DenisPreQaMaterializationError(
                f"Denis selection row {expected_rank + 1} has invalid numeric data."
            ) from error
        if (
            row.selection_rank != expected_rank
            or not row.category
            or row.source_audio_size_bytes <= 0
            or row.parent_group_id != DENIS_SINGLE_SPEAKER_GROUP
            or row.speaker_pseudo_id != DENIS_SINGLE_SPEAKER_GROUP
            or row.sample_id in sample_ids
            or row.member_stem in member_stems
            or row.literal_text_sha256 in literal_hashes
            or row.whitespace_canonical_text_sha256 in canonical_hashes
            or row.source_audio_sha256 in audio_hashes
            or not row.sample_id.startswith(f"{DENIS_SOURCE_ID}:")
            or row.text_id
            != f"{DENIS_SOURCE_ID}:text:{row.whitespace_canonical_text_sha256}"
        ):
            raise DenisPreQaMaterializationError(
                f"Denis selection row {expected_rank + 1} violates the frozen contract."
            )
        rows.append(row)
        sample_ids.add(row.sample_id)
        member_stems.add(row.member_stem)
        literal_hashes.add(row.literal_text_sha256)
        canonical_hashes.add(row.whitespace_canonical_text_sha256)
        audio_hashes.add(row.source_audio_sha256)
    category_counts = Counter(row.category for row in rows)
    if (
        len(category_counts) != 3
        or max(category_counts.values()) - min(category_counts.values()) > 1
    ):
        raise DenisPreQaMaterializationError("Denis selection categories are not balanced.")
    return tuple(rows)


def bind_denis_selection(
    selection: Sequence[FrozenDenisSelectionRow], records: Sequence[DenisRecord]
) -> tuple[BoundDenisSelectionRow, ...]:
    """Require each frozen identity/hash to match the byte-pinned archive inspection."""

    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise DenisPreQaMaterializationError("Pinned Denis archive has duplicate sample IDs.")
    bound: list[BoundDenisSelectionRow] = []
    for row in selection:
        record = by_id.get(row.sample_id)
        if record is None or (
            record.member_stem != row.member_stem
            or record.category != row.category
            or record.literal_text_sha256 != row.literal_text_sha256
            or record.whitespace_canonical_text_sha256
            != row.whitespace_canonical_text_sha256
            or record.nfkc_whitespace_canonical_text_sha256
            != row.nfkc_whitespace_canonical_text_sha256
            or record.audio_sha256 != row.source_audio_sha256
            or record.audio_size_bytes != row.source_audio_size_bytes
        ):
            raise DenisPreQaMaterializationError(
                f"Pinned archive binding changed for Denis row {row.sample_id}."
            )
        bound.append(BoundDenisSelectionRow(row, record))
    return tuple(bound)


def _safe_member(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DenisPreQaMaterializationError("Denis TAR has an unsafe member path.")


def extract_selected_denis_audio(
    archive_path: Path, bound: Sequence[BoundDenisSelectionRow], destination: Path
) -> dict[str, Path]:
    """Atomically copy only frozen source bytes, retaining the upstream .webm suffix."""

    if destination.exists() or not destination.parent.is_dir():
        raise DenisPreQaMaterializationError(
            "Denis raw destination must be new and have an existing parent."
        )
    wanted = {f"{row.record.member_stem}.webm": row for row in bound}
    if len(wanted) != len(bound):
        raise DenisPreQaMaterializationError("Denis selection maps multiple rows to one member.")
    stage = Path(tempfile.mkdtemp(prefix=".kds-denis-raw-", dir=destination.parent))
    staged_destination = stage / destination.name
    staged_destination.mkdir()
    extracted: dict[str, Path] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                _safe_member(member.name)
                row = wanted.get(member.name)
                if row is None:
                    continue
                if not member.isfile() or member.size != row.selection.source_audio_size_bytes:
                    raise DenisPreQaMaterializationError(
                        f"Selected Denis member changed: {member.name}."
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise DenisPreQaMaterializationError(
                        f"Cannot read selected Denis member: {member.name}."
                    )
                target = staged_destination / f"denis_ru_{row.selection.selection_rank:03d}.webm"
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if sha256_file(target) != row.selection.source_audio_sha256:
                    raise DenisPreQaMaterializationError(
                        f"Selected Denis audio hash changed: {member.name}."
                    )
                extracted[row.selection.sample_id] = target
        if set(extracted) != {row.selection.sample_id for row in bound}:
            raise DenisPreQaMaterializationError("Pinned archive lacks selected Denis audio.")
        staged_destination.replace(destination)
    except (OSError, tarfile.TarError) as error:
        raise DenisPreQaMaterializationError(
            f"Cannot safely extract selected Denis audio: {error}"
        ) from error
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {sample_id: destination / path.name for sample_id, path in extracted.items()}


def _relative_to_data_root(data_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(data_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise DenisPreQaMaterializationError("Denis output path escapes data root.") from error


def denis_raw_manifest_rows(
    bound: Sequence[BoundDenisSelectionRow],
    extracted: Mapping[str, Path],
    data_root: Path,
    created_at: str,
) -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    for item in bound:
        path = extracted.get(item.selection.sample_id)
        if path is None:
            raise DenisPreQaMaterializationError("Missing selected Denis extraction result.")
        rows.append(
            ManifestRow(
                sample_id=item.selection.sample_id,
                relative_path=_relative_to_data_root(data_root, path),
                sha256=item.selection.source_audio_sha256,
                split="ood",
                label="bonafide",
                language="ru",
                code_switch="false",
                parent_group_id=DENIS_SINGLE_SPEAKER_GROUP,
                source_name=DENIS_SOURCE_ID,
                source_license=DENIS_SOURCE_LICENSE,
                rights_basis=(
                    "Pinned MDC Denis 1.0 archive; CC0-1.0 card; personal research external "
                    "holdout only"
                ),
                speaker_pseudo_id=DENIS_SINGLE_SPEAKER_GROUP,
                text_id=item.selection.text_id,
                text_hash=item.selection.whitespace_canonical_text_sha256,
                duration_s=float(item.record.duration_seconds),
                generator_family="",
                generator_name="",
                generator_version="",
                voice_id="",
                clone_consent_id="",
                device="unknown",
                capture_route="denis_mdc_single_speaker_read_speech",
                original_sr=item.record.sample_rate_hz,
                codec="ogg_opus_source_suffix_webm",
                augmentation_chain="",
                augmentation_seed="",
                created_at=created_at,
            )
        )
    return tuple(rows)


def _rejection_reason(issue: PreprocessIssue) -> str:
    for reason in ("insufficient_speech", "signal_too_quiet", "excessive_clipping"):
        if reason in issue.detail:
            return reason
    return "other"


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
            raise DenisPreQaMaterializationError(
                "Denis manifest/receipt outputs must be distinct and new."
            )
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        _relative_to_data_root(data_root, arguments.raw_destination.parent)
        selection = load_frozen_denis_selection(
            arguments.selection_csv, arguments.selection_receipt, project_root
        )
        inspection = inspect_denis_archive(arguments.archive)
        bound = bind_denis_selection(selection, inspection.records)
        extracted = extract_selected_denis_audio(
            arguments.archive, bound, arguments.raw_destination
        )
        raw_rows = denis_raw_manifest_rows(
            bound, extracted, data_root, arguments.materialized_at
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
        ready_ids = {row.sample_id for row in prepared.processed_rows}
        category_by_id = {row.selection.sample_id: row.selection.category for row in bound}
        rejected_by_category = Counter(category_by_id[issue.sample_id] for issue in prepared.issues)
        ready_by_category = Counter(
            category_by_id[sample_id] for sample_id in ready_ids
        )
        ready_count = len(prepared.processed_rows)
        outcome = (
            "target_79_met"
            if ready_count >= DENIS_TARGET_PAIRS
            else "minimum_60_met_but_target_not_met"
            if ready_count >= 60
            else "stop_below_minimum_60"
        )
        with tempfile.TemporaryDirectory(
            prefix="kds-denis-materialization-", dir=arguments.materialization_receipt.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged_raw = stage / arguments.raw_manifest.name
            staged_ready = stage / arguments.ready_manifest.name
            staged_receipt = stage / arguments.materialization_receipt.name
            write_manifest(staged_raw, raw_rows)
            write_manifest(staged_ready, prepared.processed_rows)
            receipt = {
                "schema_version": 1,
                "protocol_id": "denis-1-0-mdc-pre-qa-materialization-v1",
                "materialized_at": arguments.materialized_at,
                "candidate_state": (
                    "frozen bona-fide source extraction and normal decode/QA/VAD only; no "
                    "synthesis, acoustic review, pairing, or detector inference"
                ),
                "archive": {
                    "expected_size_bytes": DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": DENIS_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_extraction": True,
                    "source_member_suffix": ".webm",
                    "decoded_container_and_codec": "Ogg/Opus",
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
                    "target_pairs": DENIS_TARGET_PAIRS,
                    "single_speaker_group": DENIS_SINGLE_SPEAKER_GROUP,
                    "post_selection_replacement_or_backfill": False,
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
                        "rows": ready_count,
                    },
                },
                "technical_qa": {
                    "pipeline": (
                        "AudioPreparationPipeline: ffprobe/ffmpeg decode, mono PCM-16 WAV "
                        "16 kHz, RMS/clipping/DC measurement, WebRTC VAD"
                    ),
                    "raw_rows": len(raw_rows),
                    "ready_rows": ready_count,
                    "rejected_rows": len(prepared.issues),
                    "ready_by_category": dict(sorted(ready_by_category.items())),
                    "rejected_by_category": dict(sorted(rejected_by_category.items())),
                    "rejection_reason_counts": dict(
                        sorted(Counter(_rejection_reason(i) for i in prepared.issues).items())
                    ),
                    "rejections": [
                        {
                            "sample_id": issue.sample_id,
                            "relative_path": issue.relative_path,
                            "category": category_by_id[issue.sample_id],
                            "reason": _rejection_reason(issue),
                            "detail": issue.detail,
                        }
                        for issue in prepared.issues
                    ],
                    "reused_rows": 0,
                    "replacement_or_backfill": False,
                },
                "target_outcome": {
                    "minimum_ready_pairs": 60,
                    "target_ready_pairs": DENIS_TARGET_PAIRS,
                    "actual_ready_pairs": ready_count,
                    "status": outcome,
                },
                "claims": {
                    "audio_extraction_performed": True,
                    "technical_decode_qa_vad_performed": True,
                    "synthetic_audio_generated": False,
                    "acoustic_review_performed": False,
                    "pairing_performed": False,
                    "detector_inference_performed": False,
                    "detector_inference_authorized": False,
                    "future_synthesis_must_use_only_ready_frozen_texts": True,
                    "external_human_source_holdout": True,
                    "generator_family_holdout_pending_candidate_synthesis": True,
                    "training_data_overlap_unverified": True,
                    "single_speaker": True,
                    "speaker_independent": False,
                    "speaker_robust": False,
                },
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in outputs):
                raise DenisPreQaMaterializationError(
                    "A Denis materialization output appeared while staging."
                )
            staged_raw.replace(arguments.raw_manifest)
            staged_ready.replace(arguments.ready_manifest)
            staged_receipt.replace(arguments.materialization_receipt)
    except (
        DenisArchiveAuditError,
        DenisPreQaMaterializationError,
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
                "target_status": outcome,
                "receipt": str(arguments.materialization_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
