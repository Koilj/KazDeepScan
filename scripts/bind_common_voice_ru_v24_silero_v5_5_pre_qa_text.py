"""Bind literal Common Voice text to the 75-row Silero V5.5 pre-QA candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
    COMMON_VOICE_RU_V24_SOURCE_ID,
    CommonVoiceIngestionError,
    CommonVoiceRecord,
    load_common_voice_metadata_from_archive,
)
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.data.silero_v5_5 import (
    SILERO_V5_5_TEXT_NORMALIZER_ID,
    SileroV55Error,
    load_silero_v5_5_runtime,
    normalize_silero_v5_5_text,
)

TEXT_BINDING_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-text-binding-v1"
MATERIALIZATION_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-materialization-v1"
ROUTE_AUDIT_PROTOCOL_ID = "silero-v5-5-ru-eugene-exact-route-audit-v1"
SELECTION_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-selection-v1"
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


class CommonVoiceSileroV55TextBindingError(ValueError):
    """Raised when the frozen candidate cannot be bound to literal source text."""


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CommonVoiceSileroV55TextBindingError(f"Cannot read {label}: {error}") from error
    if not isinstance(raw, dict):
        raise CommonVoiceSileroV55TextBindingError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], raw)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise CommonVoiceSileroV55TextBindingError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _project_file(project_root: Path, path_value: object, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise CommonVoiceSileroV55TextBindingError(f"{label} path must be non-empty.")
    try:
        candidate = (project_root / path_value).resolve(strict=True)
        candidate.relative_to(project_root)
    except (OSError, ValueError) as error:
        raise CommonVoiceSileroV55TextBindingError(
            f"{label} must be an existing file below project root."
        ) from error
    return candidate


def _receipt_file(
    project_root: Path, payload: Mapping[str, object], label: str
) -> tuple[Path, int | None]:
    path = _project_file(project_root, payload.get("path"), label)
    expected_sha256 = _sha256(payload.get("sha256"), f"{label} SHA-256")
    if sha256_file(path) != expected_sha256:
        raise CommonVoiceSileroV55TextBindingError(f"{label} bytes differ from its receipt.")
    rows = payload.get("rows")
    if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0):
        raise CommonVoiceSileroV55TextBindingError(f"{label} rows must be a positive integer.")
    return path, rows


def load_ready_pre_qa_candidate(
    *, ready_manifest: Path, materialization_receipt: Path, project_root: Path
) -> tuple[ManifestRow, ...]:
    """Load only the ready manifest pinned by the completed materialization receipt."""

    project_root = project_root.resolve(strict=True)
    ready_path = ready_manifest.resolve(strict=True)
    receipt_path = materialization_receipt.resolve(strict=True)
    try:
        ready_path.relative_to(project_root)
        receipt_path.relative_to(project_root)
    except ValueError as error:
        raise CommonVoiceSileroV55TextBindingError(
            "Ready manifest and materialization receipt must live below project root."
        ) from error
    receipt = _json_object(receipt_path, "materialization receipt")
    outputs = receipt.get("outputs")
    archive = receipt.get("archive")
    selection = receipt.get("selection")
    technical_qa = receipt.get("technical_qa")
    claims = receipt.get("claims")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != MATERIALIZATION_PROTOCOL_ID
        or not isinstance(outputs, Mapping)
        or not isinstance(archive, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(technical_qa, Mapping)
        or not isinstance(claims, Mapping)
    ):
        raise CommonVoiceSileroV55TextBindingError(
            "Materialization receipt has an invalid pre-QA contract."
        )
    ready_output = outputs.get("ready_manifest")
    raw_output = outputs.get("raw_manifest")
    if not isinstance(ready_output, Mapping) or not isinstance(raw_output, Mapping):
        raise CommonVoiceSileroV55TextBindingError(
            "Materialization receipt lacks both manifest outputs."
        )
    recorded_ready_path, ready_rows = _receipt_file(
        project_root, ready_output, "Materialization ready manifest"
    )
    _receipt_file(project_root, raw_output, "Materialization raw manifest")
    if recorded_ready_path != ready_path or ready_rows != 75:
        raise CommonVoiceSileroV55TextBindingError(
            "This text binding requires the exact 75-row ready pre-QA manifest."
        )
    if (
        archive.get("expected_size_bytes") != COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive.get("expected_sha256") != COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256
        or archive.get("identity_verified_before_metadata_read_and_extraction") is not True
        or selection.get("one_record_per_client_group") is not True
        or selection.get("post_selection_backfill") is not False
        or technical_qa.get("raw_rows") != 80
        or technical_qa.get("ready_rows") != 75
        or technical_qa.get("reused_rows") != 0
        or technical_qa.get("replacement_or_backfill") is not False
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("acoustic_review_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
        or claims.get("future_synthesis_must_use_only_ready_frozen_texts") is not True
    ):
        raise CommonVoiceSileroV55TextBindingError(
            "Materialization receipt violates the immutable pre-QA boundary."
        )
    rows = tuple(load_manifest(ready_path))
    validate_manifest(rows)
    if (
        len(rows) != 75
        or len({row.sample_id for row in rows}) != len(rows)
        or len({row.parent_group_id for row in rows}) != len(rows)
        or len({row.text_hash for row in rows}) != len(rows)
        or any(
            row.split != "test"
            or row.label != "bonafide"
            or row.language != "ru"
            or row.source_name != COMMON_VOICE_RU_V24_SOURCE_ID
            for row in rows
        )
    ):
        raise CommonVoiceSileroV55TextBindingError(
            "Ready pre-QA manifest does not contain 75 unique Common Voice RU test bona-fide rows."
        )
    return rows


def load_frozen_selection_for_text_binding(
    *, selection_csv: Path, selection_receipt: Path, project_root: Path
) -> dict[str, tuple[str, str]]:
    """Load the immutable 80-row selection only through its write-once receipt."""

    project_root = project_root.resolve(strict=True)
    selection_path = selection_csv.resolve(strict=True)
    receipt_path = selection_receipt.resolve(strict=True)
    try:
        selection_path.relative_to(project_root)
        receipt_path.relative_to(project_root)
    except ValueError as error:
        raise CommonVoiceSileroV55TextBindingError(
            "Frozen selection and its receipt must live below project root."
        ) from error
    receipt = _json_object(receipt_path, "pre-QA selection receipt")
    output = receipt.get("output_selection")
    policy = receipt.get("selection_policy")
    claims = receipt.get("claims")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != SELECTION_PROTOCOL_ID
        or not isinstance(output, Mapping)
        or not isinstance(policy, Mapping)
        or not isinstance(claims, Mapping)
        or output.get("path") != selection_csv.as_posix()
        or output.get("rows") != 80
        or _sha256(output.get("sha256"), "Selection CSV SHA-256") != sha256_file(selection_path)
        or policy.get("kind") != "seeded_two_stage_one_record_per_client_group"
        or policy.get("selected_records") != 80
        or policy.get("selected_client_groups") != 80
        or policy.get("post_selection_backfill") is not False
        or policy.get("selection_uses_audio_or_duration") is not False
        or policy.get("selection_uses_detector_or_model_output") is not False
        or policy.get("selection_uses_model_metrics_or_final_errors") is not False
        or claims.get("selection_frozen") is not True
        or claims.get("audio_extraction_performed") is not False
        or claims.get("future_extraction_must_use_only_selected_clip_names") is not True
        or claims.get("qa_rejects_must_not_trigger_backfill") is not True
    ):
        raise CommonVoiceSileroV55TextBindingError(
            "Pre-QA selection receipt has an invalid immutable-selection contract."
        )
    try:
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != SELECTION_FIELDS:
                raise CommonVoiceSileroV55TextBindingError(
                    "Frozen selection CSV has an invalid schema."
                )
            source_rows = list(reader)
    except OSError as error:
        raise CommonVoiceSileroV55TextBindingError(
            f"Cannot read frozen selection CSV: {error}"
        ) from error
    by_sample: dict[str, tuple[str, str]] = {}
    groups: set[str] = set()
    text_hashes: set[str] = set()
    for rank, row in enumerate(source_rows, start=1):
        sample_id = (row.get("sample_id") or "").strip()
        text_id = (row.get("text_id") or "").strip()
        text_hash = (row.get("text_hash") or "").strip()
        group = (row.get("parent_group_id") or "").strip()
        try:
            selected_rank = int(row.get("selection_rank") or "")
        except ValueError as error:
            raise CommonVoiceSileroV55TextBindingError(
                f"Frozen selection row {rank + 1} has an invalid rank."
            ) from error
        if (
            selected_rank != rank
            or not sample_id
            or not text_id
            or not group
            or sample_id in by_sample
            or group in groups
            or text_hash in text_hashes
            or _sha256(text_hash, f"Frozen selection row {rank + 1} text hash") != text_hash
        ):
            raise CommonVoiceSileroV55TextBindingError(
                f"Frozen selection row {rank + 1} violates the immutable contract."
            )
        by_sample[sample_id] = (text_id, text_hash)
        groups.add(group)
        text_hashes.add(text_hash)
    if len(by_sample) != 80 or len(groups) != 80 or len(text_hashes) != 80:
        raise CommonVoiceSileroV55TextBindingError(
            "Frozen selection does not retain 80 unique samples, groups, and texts."
        )
    return by_sample


def validate_route_audit(route_audit: Path, model_lock: Path) -> None:
    """Require the completed exact-route novelty audit to pin this model lock."""

    audit = _json_object(route_audit, "exact-route audit")
    model_lock_input = audit.get("model_lock")
    runtime_policy = audit.get("runtime_policy")
    route_gate = audit.get("route_gate")
    claims = audit.get("claims")
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol_id") != ROUTE_AUDIT_PROTOCOL_ID
        or not isinstance(model_lock_input, Mapping)
        or not isinstance(runtime_policy, Mapping)
        or not isinstance(route_gate, Mapping)
        or not isinstance(claims, Mapping)
    ):
        raise CommonVoiceSileroV55TextBindingError("Exact-route audit has an invalid contract.")
    if (
        model_lock_input.get("path") != model_lock.as_posix()
        or _sha256(model_lock_input.get("sha256"), "Exact-route audit model lock SHA-256")
        != sha256_file(model_lock)
        or runtime_policy.get("fixed_voice_id") != "eugene"
        or runtime_policy.get("sample_rate") != 48000
        or runtime_policy.get("reference_audio") != "forbidden"
        or runtime_policy.get("voice_cloning") is not False
        or runtime_policy.get("text_input_only") is not True
        or runtime_policy.get("ssml") != "forbidden"
        or runtime_policy.get("voice_path") != "forbidden"
        or runtime_policy.get("symbol_durs") != "forbidden"
        or route_gate.get("novelty_claim") != "unseen_exact_generator_route"
        or route_gate.get("architecture_independence_claim") is not False
        or route_gate.get("speaker_independence_claim") is not False
        or route_gate.get("exact_route_overlap_rows") != 0
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("vendor_family_independence") is not False
        or claims.get("speaker_independence") is not False
        or claims.get("reference_audio_or_voice_cloning_used") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise CommonVoiceSileroV55TextBindingError(
            "Exact-route audit does not authorize this narrow new candidate route."
        )


def bind_literal_texts(
    rows: Sequence[ManifestRow], records: Sequence[CommonVoiceRecord]
) -> tuple[dict[str, object], ...]:
    """Bind each ready manifest row to the archive's exact literal sentence."""

    records_by_sample = {
        f"{COMMON_VOICE_RU_V24_SOURCE_ID}:{Path(record.clip_name).stem}": record
        for record in records
    }
    if len(records_by_sample) != len(records):
        raise CommonVoiceSileroV55TextBindingError(
            "Common Voice test metadata repeats sample IDs."
        )
    bound: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item.sample_id):
        record = records_by_sample.get(row.sample_id)
        if record is None:
            raise CommonVoiceSileroV55TextBindingError(
                f"Pinned archive lacks ready candidate sample {row.sample_id!r}."
            )
        literal_text_sha256 = hashlib.sha256(record.sentence.encode("utf-8")).hexdigest()
        normalized = normalize_silero_v5_5_text(record.sentence)
        normalized_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if (
            record.split != "test"
            or record.sentence_id != row.text_id
            or literal_text_sha256 != row.text_hash
            or normalized != record.sentence
            or normalized_sha256 != row.text_hash
        ):
            raise CommonVoiceSileroV55TextBindingError(
                "Ready candidate text is not the exact literal V5.5-safe source text: "
                f"{row.sample_id!r}."
            )
        bound.append(
            {
                "sample_id": row.sample_id,
                "text_id": row.text_id,
                "text_hash": row.text_hash,
                "source_split": record.split,
                "literal_text_sha256": literal_text_sha256,
                "normalizer_id": SILERO_V5_5_TEXT_NORMALIZER_ID,
                "normalized_text_sha256": normalized_sha256,
            }
        )
    return tuple(bound)


def _binding_digest(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        "\n".join(
            "\t".join(
                str(row[field])
                for field in (
                    "sample_id",
                    "text_id",
                    "text_hash",
                    "literal_text_sha256",
                    "normalizer_id",
                    "normalized_text_sha256",
                )
            )
            for row in rows
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--bound-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.bound_at.replace("Z", "+00:00"))
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise CommonVoiceSileroV55TextBindingError(
                "Text-binding output must be new and its parent must exist."
            )
        project_root = arguments.project_root.resolve(strict=True)
        rows = load_ready_pre_qa_candidate(
            ready_manifest=arguments.ready_manifest,
            materialization_receipt=arguments.materialization_receipt,
            project_root=project_root,
        )
        selection_by_id = load_frozen_selection_for_text_binding(
            selection_csv=arguments.selection_csv,
            selection_receipt=arguments.selection_receipt,
            project_root=project_root,
        )
        if (
            len(selection_by_id) != 80
            or not {row.sample_id for row in rows}.issubset(selection_by_id)
            or any(
                selection_by_id[row.sample_id] != (row.text_id, row.text_hash)
                for row in rows
            )
        ):
            raise CommonVoiceSileroV55TextBindingError(
                "Ready candidate is not a non-reselected subset of the frozen 80-row selection."
            )
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1 or lock.models[0].model_id != "silero_v5_5_ru_eugene":
            raise ResearchTtsError(
                "Text binding requires exactly the locked Silero V5.5 eugene model."
            )
        runtime = load_silero_v5_5_runtime(lock.models[0])
        validate_route_audit(arguments.route_audit, arguments.model_lock)
        records = load_common_voice_metadata_from_archive(arguments.archive, ("test",))
        bound_rows = bind_literal_texts(rows, records)
        report = {
            "schema_version": 1,
            "protocol_id": TEXT_BINDING_PROTOCOL_ID,
            "bound_at": arguments.bound_at,
            "inputs": {
                "ready_manifest": {
                    "path": arguments.ready_manifest.as_posix(),
                    "sha256": sha256_file(arguments.ready_manifest),
                    "rows": len(rows),
                },
                "materialization_receipt": {
                    "path": arguments.materialization_receipt.as_posix(),
                    "sha256": sha256_file(arguments.materialization_receipt),
                },
                "selection_csv": {
                    "path": arguments.selection_csv.as_posix(),
                    "sha256": sha256_file(arguments.selection_csv),
                    "rows": len(selection_by_id),
                },
                "selection_receipt": {
                    "path": arguments.selection_receipt.as_posix(),
                    "sha256": sha256_file(arguments.selection_receipt),
                },
                "common_voice_archive": {
                    "path": str(arguments.archive),
                    "expected_size_bytes": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_metadata_read": True,
                },
                "silero_v5_5_model_lock": {
                    "path": arguments.model_lock.as_posix(),
                    "sha256": sha256_file(arguments.model_lock),
                    "model_id": lock.models[0].model_id,
                    "runtime_kind": lock.models[0].runtime["kind"],
                    "fixed_speaker": runtime.fixed_speaker,
                    "sample_rate": runtime.sample_rate,
                },
                "exact_route_audit": {
                    "path": arguments.route_audit.as_posix(),
                    "sha256": sha256_file(arguments.route_audit),
                },
            },
            "input_contract": {
                "ready_rows_only": True,
                "literal_source_text_only": True,
                "normalization": (
                    "whitespace_only_and_exact_byte_equivalent_after_archive_canonicalization"
                ),
                "external_text_normalizer_or_stress_model": "forbidden",
                "text_replacement_or_reselection": "forbidden",
                "audio_or_duration_used_for_text_binding": False,
                "detector_or_metric_used": False,
            },
            "rows": bound_rows,
            "text_binding_sha256": _binding_digest(bound_rows),
            "claims": {
                "audio_extraction_performed": False,
                "synthetic_audio_generated": False,
                "acoustic_review_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "future_synthesis_must_create_exactly_one_fixed_eugene_wav_per_bound_text": True,
                "failed_synthesis_or_qa_rows_must_not_be_replaced_or_backfilled": True,
            },
        }
        arguments.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        CommonVoiceIngestionError,
        CommonVoiceSileroV55TextBindingError,
        ManifestError,
        OSError,
        ResearchTtsError,
        SileroV55Error,
        ValueError,
    ) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "rows": len(bound_rows),
                "text_binding_sha256": report["text_binding_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
