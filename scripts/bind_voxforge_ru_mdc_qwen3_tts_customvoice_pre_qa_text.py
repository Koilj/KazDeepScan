"""Bind literal VoxForge text to the ready Qwen3-TTS CustomVoice pre-QA rows."""

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
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.data.voxforge import (
    VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
    VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
    VOXFORGE_RU_SOURCE_ID,
    VoxForgeRuAuditError,
    VoxForgeRuRecord,
    load_voxforge_ru_metadata,
)
from kds.eval.voxforge_metadata_screen import voxforge_metadata_identity

_MATERIALIZATION_PROTOCOL_ID = "voxforge-ru-mdc-pre-qa-materialization-v1"
_ROUTE_AUDIT_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-exact-route-audit-v1"
_SELECTION_PROTOCOL_ID = "voxforge-ru-mdc-pre-qa-selection-v1"
_TEXT_BINDING_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-text-binding-v1"
_MODEL_ID = "qwen3_tts_0_6b_customvoice_aiden_q8_0"
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


class VoxForgeQwenTextBindingError(ValueError):
    """Raised when frozen VoxForge text cannot safely bind to this exact route."""


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgeQwenTextBindingError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise VoxForgeQwenTextBindingError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise VoxForgeQwenTextBindingError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _project_file(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise VoxForgeQwenTextBindingError(f"{label} path must be non-empty.")
    try:
        path = (project_root / value).resolve(strict=True)
        path.relative_to(project_root)
    except (OSError, ValueError) as error:
        raise VoxForgeQwenTextBindingError(
            f"{label} must be an existing file beneath project root."
        ) from error
    return path


def _receipt_file(
    project_root: Path, payload: object, label: str
) -> tuple[Path, int | None]:
    if not isinstance(payload, Mapping):
        raise VoxForgeQwenTextBindingError(f"{label} binding is absent.")
    path = _project_file(project_root, payload.get("path"), label)
    if sha256_file(path) != _sha256(payload.get("sha256"), f"{label} SHA-256"):
        raise VoxForgeQwenTextBindingError(f"{label} bytes differ from its receipt.")
    rows = payload.get("rows")
    if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0):
        raise VoxForgeQwenTextBindingError(f"{label} rows must be a positive integer.")
    return path, rows


def _load_selection(
    selection_csv: Path, selection_receipt: Path, project_root: Path
) -> dict[str, dict[str, object]]:
    selection_path = selection_csv.resolve(strict=True)
    receipt_path = selection_receipt.resolve(strict=True)
    try:
        selection_path.relative_to(project_root)
        receipt_path.relative_to(project_root)
    except ValueError as error:
        raise VoxForgeQwenTextBindingError(
            "Frozen selection and receipt must live beneath project root."
        ) from error
    receipt = _json_object(receipt_path, "frozen VoxForge selection receipt")
    output = receipt.get("output_selection")
    policy = receipt.get("selection_policy")
    claims = receipt.get("claims")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != _SELECTION_PROTOCOL_ID
        or not isinstance(output, Mapping)
        or output.get("path") != selection_csv.as_posix()
        or output.get("rows") != 81
        or _sha256(output.get("sha256"), "Selection CSV SHA-256") != sha256_file(selection_path)
        or not isinstance(policy, Mapping)
        or policy.get("selected_records") != 81
        or policy.get("selected_contributor_groups") != 81
        or policy.get("selected_canonical_prompt_text_groups") != 81
        or policy.get("post_selection_backfill") is not False
        or policy.get("selection_uses_audio_or_duration") is not False
        or policy.get("selection_uses_detector_or_model_output") is not False
        or not isinstance(claims, Mapping)
        or claims.get("selection_frozen") is not True
        or claims.get("qa_rejects_must_not_trigger_backfill") is not True
    ):
        raise VoxForgeQwenTextBindingError("Frozen selection receipt contract is invalid.")
    try:
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != _SELECTION_FIELDS:
                raise VoxForgeQwenTextBindingError("Frozen selection CSV schema is invalid.")
            source_rows = list(reader)
    except OSError as error:
        raise VoxForgeQwenTextBindingError(f"Cannot read frozen selection CSV: {error}") from error
    if len(source_rows) != 81:
        raise VoxForgeQwenTextBindingError("Frozen selection CSV must contain 81 rows.")
    result: dict[str, dict[str, object]] = {}
    groups: set[str] = set()
    text_hashes: set[str] = set()
    for expected_rank, row in enumerate(source_rows, start=1):
        sample_id = (row.get("sample_id") or "").strip()
        try:
            rank = int(row.get("selection_rank") or "")
        except ValueError as error:
            raise VoxForgeQwenTextBindingError(
                f"Frozen selection row {expected_rank + 1} rank is invalid."
            ) from error
        item: dict[str, object] = {
            "selection_rank": rank,
            "sample_id": sample_id,
            "submission_pseudo_id": (row.get("submission_pseudo_id") or "").strip(),
            "prompt_id": (row.get("prompt_id") or "").strip(),
            "parent_group_id": (row.get("parent_group_id") or "").strip(),
            "speaker_pseudo_id": (row.get("speaker_pseudo_id") or "").strip(),
            "prompt_text_hash": _sha256(row.get("prompt_text_hash"), "prompt text hash"),
            "original_prompt_text_hash": _sha256(
                row.get("original_prompt_text_hash"), "original prompt text hash"
            ),
        }
        if (
            rank != expected_rank
            or not sample_id
            or sample_id in result
            or not item["prompt_id"]
            or item["parent_group_id"] != item["speaker_pseudo_id"]
            or item["parent_group_id"] in groups
            or item["prompt_text_hash"] in text_hashes
        ):
            raise VoxForgeQwenTextBindingError(
                f"Frozen selection row {expected_rank + 1} violates its contract."
            )
        result[sample_id] = item
        groups.add(cast(str, item["parent_group_id"]))
        text_hashes.add(cast(str, item["prompt_text_hash"]))
    return result


def load_ready_candidate(
    ready_manifest: Path, materialization_receipt: Path, project_root: Path
) -> tuple[ManifestRow, ...]:
    """Load only the 79 ready rows pinned by the completed materialization receipt."""

    ready_path = ready_manifest.resolve(strict=True)
    receipt_path = materialization_receipt.resolve(strict=True)
    try:
        ready_path.relative_to(project_root)
        receipt_path.relative_to(project_root)
    except ValueError as error:
        raise VoxForgeQwenTextBindingError(
            "Ready manifest and receipt must live beneath project root."
        ) from error
    receipt = _json_object(receipt_path, "VoxForge materialization receipt")
    outputs = receipt.get("outputs")
    archive = receipt.get("archive")
    selection = receipt.get("selection")
    technical_qa = receipt.get("technical_qa")
    claims = receipt.get("claims")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != _MATERIALIZATION_PROTOCOL_ID
        or not isinstance(outputs, Mapping)
        or not isinstance(archive, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(technical_qa, Mapping)
        or not isinstance(claims, Mapping)
    ):
        raise VoxForgeQwenTextBindingError("Materialization receipt contract is invalid.")
    recorded_ready, ready_rows = _receipt_file(
        project_root, outputs.get("ready_manifest"), "Materialization ready manifest"
    )
    _receipt_file(project_root, outputs.get("raw_manifest"), "Materialization raw manifest")
    if (
        recorded_ready != ready_path
        or ready_rows != 79
        or archive.get("expected_size_bytes") != VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive.get("expected_sha256") != VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256
        or archive.get("identity_verified_before_metadata_read_and_extraction") is not True
        or selection.get("one_record_per_prompt_text_and_contributor_group") is not True
        or selection.get("post_selection_backfill") is not False
        or technical_qa.get("raw_rows") != 81
        or technical_qa.get("ready_rows") != 79
        or not isinstance(technical_qa.get("rejected_rows"), list)
        or len(technical_qa["rejected_rows"]) != 2
        or technical_qa.get("replacement_or_backfill") is not False
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("pairing_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
        or claims.get("future_synthesis_must_use_only_ready_frozen_texts") is not True
    ):
        raise VoxForgeQwenTextBindingError("Materialization receipt violates the pre-QA boundary.")
    rows = tuple(load_manifest(ready_path))
    validate_manifest(rows)
    if (
        len(rows) != 79
        or len({row.sample_id for row in rows}) != len(rows)
        or len({row.parent_group_id for row in rows}) != len(rows)
        or len({row.text_hash for row in rows}) != len(rows)
        or any(
            row.split != "test"
            or row.label != "bonafide"
            or row.language != "ru"
            or row.source_name != VOXFORGE_RU_SOURCE_ID
            or row.source_license != "GPL-3.0-or-later"
            for row in rows
        )
    ):
        raise VoxForgeQwenTextBindingError(
            "Ready candidate must be 79 unique VoxForge RU bona-fide test rows."
        )
    return rows


def validate_route(route_audit: Path, model_lock: Path, artifact_lock: Path) -> None:
    """Require the accepted route/audit and artifact lock for this exact local model lock."""

    audit = _json_object(route_audit, "Qwen exact-route audit")
    audit_lock = audit.get("model_lock")
    runtime = audit.get("runtime_policy")
    gate = audit.get("route_gate")
    claims = audit.get("claims")
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol_id") != _ROUTE_AUDIT_PROTOCOL_ID
        or not isinstance(audit_lock, Mapping)
        or audit_lock.get("path") != model_lock.as_posix()
        or _sha256(audit_lock.get("sha256"), "Route audit model lock SHA-256")
        != sha256_file(model_lock)
        or not isinstance(runtime, Mapping)
        or runtime.get("fixed_voice_id") != "qwen3_tts_customvoice:aiden"
        or runtime.get("fixed_speaker_name") != "aiden"
        or runtime.get("target_language") != "ru"
        or runtime.get("sample_rate") != 24000
        or runtime.get("reference_audio") != "forbidden"
        or runtime.get("voice_cloning") is not False
        or runtime.get("voice_design") != "forbidden"
        or runtime.get("text_input_only") is not True
        or runtime.get("runtime_auto_download") != "forbidden"
        or not isinstance(gate, Mapping)
        or gate.get("novelty_claim") != "unseen_exact_generator_route"
        or gate.get("exact_route_overlap_rows") != 0
        or gate.get("architecture_independence_claim") is not False
        or gate.get("speaker_independence_claim") is not False
        or not isinstance(claims, Mapping)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("reference_audio_or_voice_cloning_used") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise VoxForgeQwenTextBindingError("Exact-route audit does not authorize this binding.")
    artifact = _json_object(artifact_lock, "Qwen artifact lock")
    model = artifact.get("model_lock")
    artifact_claims = artifact.get("claims")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("source_id") != "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1"
        or not isinstance(model, Mapping)
        or model.get("path") != model_lock.as_posix()
        or _sha256(model.get("sha256"), "Artifact lock model SHA-256") != sha256_file(model_lock)
        or not isinstance(artifact_claims, Mapping)
        or artifact_claims.get("artifact_lock_passed") is not True
        or artifact_claims.get("synthesis_performed") is not False
    ):
        raise VoxForgeQwenTextBindingError("Artifact lock does not authorize this binding.")


def bind_literal_texts(
    rows: Sequence[ManifestRow],
    records: Sequence[VoxForgeRuRecord],
    selection: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Bind ready rows to canonical archive transcript text without persisting transcript text."""

    by_sample = {voxforge_metadata_identity(record).sample_id: record for record in records}
    if len(by_sample) != len(records):
        raise VoxForgeQwenTextBindingError("Pinned archive has duplicate VoxForge sample IDs.")
    bound: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item.sample_id):
        source = by_sample.get(row.sample_id)
        selected = selection.get(row.sample_id)
        if source is None or selected is None:
            raise VoxForgeQwenTextBindingError(
                f"Pinned archive lacks ready candidate {row.sample_id!r}."
            )
        identity = voxforge_metadata_identity(source)
        literal_text = source.prompt_text
        literal_sha256 = hashlib.sha256(literal_text.encode("utf-8")).hexdigest()
        original_sha256 = hashlib.sha256(source.original_prompt_text.encode("utf-8")).hexdigest()
        encoded = literal_text.encode("utf-8")
        if (
            source.prompt_id != row.text_id
            or literal_sha256 != row.text_hash
            or literal_sha256 != selected.get("prompt_text_hash")
            or original_sha256 != selected.get("original_prompt_text_hash")
            or identity.parent_group_id != row.parent_group_id
            or identity.speaker_pseudo_id != row.speaker_pseudo_id
            or not literal_text
            or "\x00" in literal_text
            or len(encoded) > 4096
        ):
            raise VoxForgeQwenTextBindingError(
                f"Ready row is not the exact Qwen-safe archive text: {row.sample_id!r}."
            )
        bound.append(
            {
                "selection_rank": selected["selection_rank"],
                "sample_id": row.sample_id,
                "text_id": row.text_id,
                "text_hash": row.text_hash,
                "original_prompt_text_hash": original_sha256,
                "source_prompt_canonicalization": "archive_PROMPTS_whitespace_collapsed",
                "literal_text_sha256": literal_sha256,
                "literal_text_utf8_bytes": len(encoded),
                "rng_seed": int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big"),
            }
        )
    return tuple(bound)


def _binding_digest(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        "\n".join(
            "\t".join(
                str(row[field])
                for field in (
                    "selection_rank",
                    "sample_id",
                    "text_id",
                    "text_hash",
                    "original_prompt_text_hash",
                    "literal_text_sha256",
                    "literal_text_utf8_bytes",
                    "rng_seed",
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
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--bound-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.bound_at.replace("Z", "+00:00"))
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise VoxForgeQwenTextBindingError(
                "Text-binding output must be new with an existing parent directory."
            )
        project_root = arguments.project_root.resolve(strict=True)
        ready_rows = load_ready_candidate(
            arguments.ready_manifest, arguments.materialization_receipt, project_root
        )
        selection = _load_selection(
            arguments.selection_csv, arguments.selection_receipt, project_root
        )
        if (
            not {row.sample_id for row in ready_rows}.issubset(selection)
            or any(
                selection[row.sample_id]["prompt_id"] != row.text_id
                or selection[row.sample_id]["prompt_text_hash"] != row.text_hash
                for row in ready_rows
            )
        ):
            raise VoxForgeQwenTextBindingError(
                "Ready rows are not an unreselected subset of the frozen selection."
            )
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1 or lock.models[0].model_id != _MODEL_ID:
            raise ResearchTtsError("Text binding requires the one pinned Qwen CustomVoice model.")
        validate_route(arguments.route_audit, arguments.model_lock, arguments.artifact_lock)
        records = load_voxforge_ru_metadata(arguments.archive)
        bound_rows = bind_literal_texts(ready_rows, records, selection)
        report = {
            "schema_version": 1,
            "protocol_id": _TEXT_BINDING_PROTOCOL_ID,
            "bound_at": arguments.bound_at,
            "inputs": {
                "ready_manifest": {
                    "path": arguments.ready_manifest.as_posix(),
                    "sha256": sha256_file(arguments.ready_manifest),
                    "rows": len(ready_rows),
                },
                "materialization_receipt": {
                    "path": arguments.materialization_receipt.as_posix(),
                    "sha256": sha256_file(arguments.materialization_receipt),
                },
                "selection_csv": {
                    "path": arguments.selection_csv.as_posix(),
                    "sha256": sha256_file(arguments.selection_csv),
                    "rows": len(selection),
                },
                "selection_receipt": {
                    "path": arguments.selection_receipt.as_posix(),
                    "sha256": sha256_file(arguments.selection_receipt),
                },
                "voxforge_archive": {
                    "path": str(arguments.archive),
                    "expected_size_bytes": VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_metadata_read": True,
                },
                "model_lock": {
                    "path": arguments.model_lock.as_posix(),
                    "sha256": sha256_file(arguments.model_lock),
                    "model_id": lock.models[0].model_id,
                    "runtime_kind": lock.models[0].runtime["kind"],
                },
                "artifact_lock": {
                    "path": arguments.artifact_lock.as_posix(),
                    "sha256": sha256_file(arguments.artifact_lock),
                },
                "exact_route_audit": {
                    "path": arguments.route_audit.as_posix(),
                    "sha256": sha256_file(arguments.route_audit),
                },
            },
            "input_contract": {
                "ready_rows_only": True,
                "literal_source_text_only": True,
                "archive_canonicalization": "PROMPTS whitespace collapse only",
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
                "pairing_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "future_synthesis_must_create_exactly_one_fixed_aiden_wav_per_bound_text": True,
                "failed_synthesis_or_qa_rows_must_not_be_replaced_or_backfilled": True,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (
        ManifestError,
        OSError,
        ResearchTtsError,
        VoxForgeQwenTextBindingError,
        VoxForgeRuAuditError,
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
