"""Create at most one fixed-Aiden WAV for every bound VoxForge RU pre-QA text."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.qwen3_tts_customvoice import (
    Qwen3TtsCustomVoiceError,
    load_qwen3_tts_customvoice,
)
from kds.data.qwen3_tts_customvoice_candidate import (
    QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID,
    Qwen3TtsCustomVoiceCandidateError,
    qwen3_tts_customvoice_spoof_row,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.data.voxforge import (
    VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
    VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
    VoxForgeRuAuditError,
    VoxForgeRuRecord,
    load_voxforge_ru_metadata,
)
from kds.eval.voxforge_metadata_screen import voxforge_metadata_identity

_SYNTHESIS_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-synthesis-v1"
_ROUTE_AUDIT_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-exact-route-audit-v1"
_TEXT_BINDING_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-text-binding-v1"
_MODEL_ID = "qwen3_tts_0_6b_customvoice_aiden_q8_0"
_HEX = frozenset("0123456789abcdef")


class VoxForgeQwenSynthesisError(ValueError):
    """Raised when this irreversible one-shot synthesis contract is violated."""


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgeQwenSynthesisError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise VoxForgeQwenSynthesisError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise VoxForgeQwenSynthesisError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_binding_input(
    inputs: Mapping[str, object], name: str, path: Path, *, rows: int | None = None
) -> None:
    value = inputs.get(name)
    if not isinstance(value, Mapping):
        raise VoxForgeQwenSynthesisError(f"Text binding lacks {name} input.")
    if (
        value.get("path") != path.as_posix()
        or _sha256(value.get("sha256"), f"Text binding {name} SHA-256") != sha256_file(path)
        or (rows is not None and value.get("rows") != rows)
    ):
        raise VoxForgeQwenSynthesisError(f"Text binding {name} no longer matches its input.")


def _binding_digest(rows: Sequence[Mapping[str, object]]) -> str:
    fields = (
        "selection_rank",
        "sample_id",
        "text_id",
        "text_hash",
        "original_prompt_text_hash",
        "literal_text_sha256",
        "literal_text_utf8_bytes",
        "rng_seed",
    )
    return hashlib.sha256(
        "\n".join("\t".join(str(row[field]) for field in fields) for row in rows).encode("utf-8")
    ).hexdigest()


def require_text_binding(
    path: Path,
    *,
    ready_manifest: Path,
    archive: Path,
    model_lock: Path,
    artifact_lock: Path,
    route_audit: Path,
) -> dict[str, Mapping[str, object]]:
    """Accept only the completed exact 79-row literal binding for this local route."""

    binding = _json_object(path, "VoxForge Qwen literal text binding")
    inputs = binding.get("inputs")
    contract = binding.get("input_contract")
    claims = binding.get("claims")
    rows = binding.get("rows")
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol_id") != _TEXT_BINDING_PROTOCOL_ID
        or not isinstance(inputs, Mapping)
        or not isinstance(contract, Mapping)
        or not isinstance(claims, Mapping)
        or not isinstance(rows, list)
        or len(rows) != 79
    ):
        raise VoxForgeQwenSynthesisError("Literal text binding contract is invalid.")
    _require_binding_input(inputs, "ready_manifest", ready_manifest, rows=79)
    _require_binding_input(inputs, "model_lock", model_lock)
    _require_binding_input(inputs, "artifact_lock", artifact_lock)
    _require_binding_input(inputs, "exact_route_audit", route_audit)
    archive_input = inputs.get("voxforge_archive")
    if (
        not isinstance(archive_input, Mapping)
        or archive_input.get("path") != str(archive)
        or archive_input.get("expected_size_bytes") != VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive_input.get("expected_sha256") != VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256
        or archive_input.get("identity_verified_before_metadata_read") is not True
    ):
        raise VoxForgeQwenSynthesisError("Text binding does not pin the expected VoxForge archive.")
    if (
        contract.get("ready_rows_only") is not True
        or contract.get("literal_source_text_only") is not True
        or contract.get("external_text_normalizer_or_stress_model") != "forbidden"
        or contract.get("text_replacement_or_reselection") != "forbidden"
        or contract.get("audio_or_duration_used_for_text_binding") is not False
        or contract.get("detector_or_metric_used") is not False
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("pairing_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
        or claims.get("future_synthesis_must_create_exactly_one_fixed_aiden_wav_per_bound_text")
        is not True
        or claims.get("failed_synthesis_or_qa_rows_must_not_be_replaced_or_backfilled")
        is not True
    ):
        raise VoxForgeQwenSynthesisError("Text binding does not retain synthesis governance.")
    bound: dict[str, Mapping[str, object]] = {}
    text_hashes: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise VoxForgeQwenSynthesisError(f"Text binding row {index} is invalid.")
        sample_id = row.get("sample_id")
        text_hash = row.get("text_hash")
        if (
            not isinstance(sample_id, str)
            or sample_id in bound
            or text_hash in text_hashes
            or _sha256(text_hash, f"Text binding row {index} text hash") != text_hash
            or row.get("literal_text_sha256") != text_hash
            or not isinstance(row.get("literal_text_utf8_bytes"), int)
            or row["literal_text_utf8_bytes"] <= 0
            or row["literal_text_utf8_bytes"] > 4096
            or not isinstance(row.get("rng_seed"), int)
            or row["rng_seed"] < 0
            or row["rng_seed"] > 2**32 - 1
        ):
            raise VoxForgeQwenSynthesisError(
                f"Text binding row {index} violates the literal one-to-one contract."
            )
        bound[sample_id] = row
        text_hashes.add(text_hash)
    if binding.get("text_binding_sha256") != _binding_digest(rows):
        raise VoxForgeQwenSynthesisError("Text binding digest differs from its rows.")
    return bound


def require_route(route_audit: Path, model_lock: Path, artifact_lock: Path) -> None:
    """Require the already accepted exact route without broadening its novelty claim."""

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
        raise VoxForgeQwenSynthesisError("Exact-route audit does not authorize synthesis.")
    artifact = _json_object(artifact_lock, "Qwen artifact lock")
    model = artifact.get("model_lock")
    artifact_claims = artifact.get("claims")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("source_id") != QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
        or not isinstance(model, Mapping)
        or model.get("path") != model_lock.as_posix()
        or _sha256(model.get("sha256"), "Artifact lock model SHA-256") != sha256_file(model_lock)
        or not isinstance(artifact_claims, Mapping)
        or artifact_claims.get("artifact_lock_passed") is not True
        or artifact_claims.get("synthesis_performed") is not False
    ):
        raise VoxForgeQwenSynthesisError("Artifact lock does not authorize synthesis.")


def source_texts(
    archive: Path, base_rows: Sequence[ManifestRow], binding: Mapping[str, Mapping[str, object]]
) -> dict[str, str]:
    """Re-read only exact canonical source text and prove it still matches the binding."""

    records = load_voxforge_ru_metadata(archive)
    by_sample = {voxforge_metadata_identity(record).sample_id: record for record in records}
    if len(by_sample) != len(records):
        raise VoxForgeQwenSynthesisError("Pinned VoxForge archive has duplicate sample IDs.")
    texts: dict[str, str] = {}
    for base in base_rows:
        record: VoxForgeRuRecord | None = by_sample.get(base.sample_id)
        bound = binding.get(base.sample_id)
        if record is None or bound is None:
            raise VoxForgeQwenSynthesisError(
                f"Pinned archive or text binding lacks {base.sample_id!r}."
            )
        literal = record.prompt_text
        literal_hash = hashlib.sha256(literal.encode("utf-8")).hexdigest()
        original_hash = hashlib.sha256(record.original_prompt_text.encode("utf-8")).hexdigest()
        seed = int.from_bytes(hashlib.sha256(literal.encode("utf-8")).digest()[:4], "big")
        if (
            record.prompt_id != base.text_id
            or literal_hash != base.text_hash
            or literal_hash != bound.get("literal_text_sha256")
            or original_hash != bound.get("original_prompt_text_hash")
            or len(literal.encode("utf-8")) != bound.get("literal_text_utf8_bytes")
            or seed != bound.get("rng_seed")
        ):
            raise VoxForgeQwenSynthesisError(
                f"Literal source text no longer matches {base.sample_id!r}."
            )
        texts[base.sample_id] = literal
    return texts


def _relative_to_data_root(data_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise VoxForgeQwenSynthesisError(
            "Synthetic asset path must stay below data root."
        ) from error


def _validate_base_rows(rows: Sequence[ManifestRow]) -> None:
    if (
        len(rows) != 79
        or len({row.sample_id for row in rows}) != 79
        or len({row.parent_group_id for row in rows}) != 79
        or len({row.text_hash for row in rows}) != 79
        or any(
            row.split != "test"
            or row.label != "bonafide"
            or row.language != "ru"
            or row.source_name != "voxforge_ru_mdc_2026_05"
            for row in rows
        )
    ):
        raise VoxForgeQwenSynthesisError(
            "Synthesis base must be exactly 79 unique frozen VoxForge RU rows."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--text-binding", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        outputs = (arguments.output_manifest, arguments.output_report)
        if (
            arguments.output_directory.exists()
            or len(set(outputs)) != len(outputs)
            or any(path.exists() or not path.parent.is_dir() for path in outputs)
        ):
            raise VoxForgeQwenSynthesisError(
                "Synthesis output directory, manifest and report must all be distinct and new."
            )
        data_root = arguments.data_root.resolve(strict=True)
        _relative_to_data_root(data_root, arguments.output_directory.parent)
        base_rows = tuple(load_manifest(arguments.base_manifest))
        validate_manifest(base_rows)
        _validate_base_rows(base_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        require_valid_assets(base_rows, data_root)
        if QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID not in ledger:
            raise LicenseLedgerError(
                ["Qwen3-TTS synthetic source is absent from the license ledger."]
            )
        binding = require_text_binding(
            arguments.text_binding,
            ready_manifest=arguments.base_manifest,
            archive=arguments.archive,
            model_lock=arguments.model_lock,
            artifact_lock=arguments.artifact_lock,
            route_audit=arguments.route_audit,
        )
        if set(binding) != {row.sample_id for row in base_rows}:
            raise VoxForgeQwenSynthesisError("Text binding does not cover exactly the ready rows.")
        require_route(arguments.route_audit, arguments.model_lock, arguments.artifact_lock)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1 or lock.models[0].model_id != _MODEL_ID:
            raise ResearchTtsError("Synthesis requires the one pinned Qwen CustomVoice model.")
        model = lock.models[0]
        runtime = load_qwen3_tts_customvoice(arguments.model_root, model)
        texts = source_texts(arguments.archive, base_rows, binding)
        arguments.output_directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".kds-voxforge-qwen3-assets-", dir=arguments.output_directory.parent
        ) as stage_name, tempfile.TemporaryDirectory(
            prefix="kds-voxforge-qwen3-metadata-", dir=arguments.output_report.parent
        ) as metadata_name:
            stage = Path(stage_name)
            stage_assets = stage / "assets"
            stage_assets.mkdir()
            rows: list[ManifestRow] = []
            staged_rows: list[ManifestRow] = []
            generated: list[dict[str, object]] = []
            failed: list[dict[str, object]] = []
            ordered_base_rows = sorted(base_rows, key=lambda item: item.sample_id)
            for index, base in enumerate(ordered_base_rows, start=1):
                file_key = hashlib.sha256(base.sample_id.encode("utf-8")).hexdigest()[:20]
                filename = f"{file_key}-{base.text_hash[:12]}.wav"
                attempt_directory = stage / f"attempt-{index:03d}"
                attempt_directory.mkdir()
                attempt_output = attempt_directory / filename
                try:
                    prepared = runtime.prepare_text(texts[base.sample_id])
                    if prepared.seed != binding[base.sample_id].get("rng_seed"):
                        raise VoxForgeQwenSynthesisError("Prepared text seed differs from binding.")
                    runtime.synthesize_to_file(prepared, attempt_output)
                    staged_output = stage_assets / filename
                    attempt_output.replace(staged_output)
                    final_output = arguments.output_directory / filename
                    final_row = qwen3_tts_customvoice_spoof_row(
                        base_row=base,
                        model=model,
                        runtime=runtime,
                        prepared=prepared,
                        relative_path=_relative_to_data_root(data_root, final_output),
                        sha256=sha256_file(staged_output),
                        duration_s=float(runtime.sample_rate and 0),
                        created_at=arguments.created_at,
                    )
                    info = sf.info(staged_output)
                    if (
                        info.samplerate != runtime.sample_rate
                        or info.channels != 1
                        or info.frames <= 0
                        or info.format != "WAV"
                    ):
                        raise VoxForgeQwenSynthesisError(
                            "Pinned Qwen3-TTS output is not non-empty mono WAV at the locked rate."
                        )
                    final_row = replace(final_row, duration_s=float(info.duration))
                    stage_row = replace(
                        final_row,
                        relative_path=_relative_to_data_root(data_root, staged_output),
                    )
                    rows.append(final_row)
                    staged_rows.append(stage_row)
                    generated.append(
                        {
                            "base_sample_id": base.sample_id,
                            "spoof_sample_id": final_row.sample_id,
                            "text_id": final_row.text_id,
                            "text_hash": final_row.text_hash,
                            "relative_path": final_row.relative_path,
                            "audio_sha256": final_row.sha256,
                            "duration_s": final_row.duration_s,
                            "rng_seed": prepared.seed,
                        }
                    )
                except (
                    OSError,
                    Qwen3TtsCustomVoiceError,
                    RuntimeError,
                    VoxForgeQwenSynthesisError,
                ) as error:
                    failed.append(
                        {
                            "base_sample_id": base.sample_id,
                            "text_id": base.text_id,
                            "text_hash": base.text_hash,
                            "error_type": type(error).__name__,
                            "detail": str(error)[-1200:],
                        }
                    )
                if index % 5 == 0 or index == len(base_rows):
                    print(
                        json.dumps(
                            {
                                "status": "running",
                                "attempted_rows": index,
                                "generated_rows": len(rows),
                                "failed_rows": len(failed),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if not rows:
                raise VoxForgeQwenSynthesisError("Every one-shot synthesis attempt failed.")
            if len(rows) + len(failed) != len(base_rows):
                raise VoxForgeQwenSynthesisError("Synthesis attempts do not cover every bound row.")
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            validate_manifest(staged_rows)
            validate_manifest_licenses(staged_rows, ledger)
            require_valid_assets(staged_rows, data_root)
            metadata_stage = Path(metadata_name)
            staged_manifest = metadata_stage / arguments.output_manifest.name
            staged_report = metadata_stage / arguments.output_report.name
            write_manifest(staged_manifest, rows)
            report = {
                "schema_version": 1,
                "protocol_id": _SYNTHESIS_PROTOCOL_ID,
                "created_at": arguments.created_at,
                "base_manifest": {
                    "path": arguments.base_manifest.as_posix(),
                    "sha256": sha256_file(arguments.base_manifest),
                    "rows": len(base_rows),
                },
                "text_binding": {
                    "path": arguments.text_binding.as_posix(),
                    "sha256": sha256_file(arguments.text_binding),
                    "rows": len(binding),
                },
                "route_audit": {
                    "path": arguments.route_audit.as_posix(),
                    "sha256": sha256_file(arguments.route_audit),
                },
                "artifact_lock": {
                    "path": arguments.artifact_lock.as_posix(),
                    "sha256": sha256_file(arguments.artifact_lock),
                },
                "model_lock": {
                    "path": arguments.model_lock.as_posix(),
                    "sha256": sha256_file(arguments.model_lock),
                    "model_id": model.model_id,
                    "fixed_speaker": runtime.fixed_speaker_name,
                    "sample_rate": runtime.sample_rate,
                },
                "output_manifest": {
                    "path": arguments.output_manifest.as_posix(),
                    "sha256": sha256_file(staged_manifest),
                    "rows": len(rows),
                },
                "generation_policy": {
                    "device": "cuda:0",
                    "fixed_profile": "qwen3_tts_customvoice:aiden",
                    "exactly_one_synthetic_per_bound_base": True,
                    "exactly_one_attempt_per_bound_text": True,
                    "successful_one_to_one_synthetic_rows": len(rows),
                    "failed_attempt_rows": len(failed),
                    "reference_audio_or_voice_cloning_used": False,
                    "voice_design_used": False,
                    "post_selection_replacement_or_backfill": False,
                    "resynthesis_after_failure": "forbidden",
                },
                "claims": {
                    "synthetic_audio_generated": True,
                    "technical_decode_qa_vad_performed": False,
                    "acoustic_review_performed": False,
                    "binary_pairing_performed": False,
                    "detector_inference_performed": False,
                    "detector_inference_authorized": False,
                },
                "generated": generated,
                "failed_attempts": failed,
            }
            staged_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if arguments.output_directory.exists() or any(path.exists() for path in outputs):
                raise VoxForgeQwenSynthesisError("A synthesis output appeared while staging.")
            stage_assets.replace(arguments.output_directory)
            require_valid_assets(rows, data_root)
            staged_manifest.replace(arguments.output_manifest)
            staged_report.replace(arguments.output_report)
    except (
        LicenseLedgerError,
        ManifestError,
        OSError,
        Qwen3TtsCustomVoiceCandidateError,
        Qwen3TtsCustomVoiceError,
        ResearchTtsError,
        VoxForgeQwenSynthesisError,
        VoxForgeRuAuditError,
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
                "attempted_rows": len(base_rows),
                "generated_rows": len(rows),
                "failed_rows": len(failed),
                "manifest": str(arguments.output_manifest),
                "report": str(arguments.output_report),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
