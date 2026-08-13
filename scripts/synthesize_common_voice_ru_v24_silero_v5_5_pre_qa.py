"""Create one irreversible fixed-eugene WAV for every bound V5.5 pre-QA text."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
    COMMON_VOICE_RU_V24_SOURCE_ID,
    CommonVoiceIngestionError,
    load_common_voice_metadata_from_archive,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.research_tts import (
    ResearchTtsError,
    ResearchTtsModel,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.data.silero_v5_5 import (
    SILERO_V5_5_FIXED_SPEAKER,
    SILERO_V5_5_SAMPLE_RATE,
    SILERO_V5_5_SOURCE_ID,
    SileroV55Error,
    SileroV55Runtime,
    load_silero_v5_5_model,
    load_silero_v5_5_runtime,
    normalize_silero_v5_5_text,
    silero_v5_5_spoof_row,
    synthesize_silero_v5_5,
)

SYNTHESIS_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-synthesis-v1"
TEXT_BINDING_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-text-binding-v1"
ROUTE_AUDIT_PROTOCOL_ID = "silero-v5-5-ru-eugene-exact-route-audit-v1"
DEVICE_ID = "local_cpu_silero_v5_5_ru_eugene"
_HEX = frozenset("0123456789abcdef")


class CommonVoiceSileroV55SynthesisError(ValueError):
    """Raised when the write-once pre-QA synthesis contract is violated."""


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CommonVoiceSileroV55SynthesisError(f"Cannot read {label}: {error}") from error
    if not isinstance(raw, dict):
        raise CommonVoiceSileroV55SynthesisError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], raw)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise CommonVoiceSileroV55SynthesisError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_binding_input(
    inputs: Mapping[str, object],
    name: str,
    path: Path,
    *,
    rows: int | None = None,
) -> None:
    value = inputs.get(name)
    if not isinstance(value, Mapping):
        raise CommonVoiceSileroV55SynthesisError(f"Text binding lacks {name} input.")
    if value.get("path") != path.as_posix() or _sha256(
        value.get("sha256"), f"Text binding {name} SHA-256"
    ) != sha256_file(path):
        raise CommonVoiceSileroV55SynthesisError(
            f"Text binding {name} is no longer the pinned input."
        )
    if rows is not None and value.get("rows") != rows:
        raise CommonVoiceSileroV55SynthesisError(
            f"Text binding {name} has an unexpected row count."
        )


def require_text_binding(
    path: Path,
    *,
    base_manifest: Path,
    archive: Path,
    model_lock: Path,
    route_audit: Path,
) -> dict[str, Mapping[str, object]]:
    """Accept only the completed literal, exact-route-bound 75-row receipt."""

    binding = _json_object(path, "pre-QA literal text binding")
    inputs = binding.get("inputs")
    contract = binding.get("input_contract")
    claims = binding.get("claims")
    rows = binding.get("rows")
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol_id") != TEXT_BINDING_PROTOCOL_ID
        or not isinstance(inputs, Mapping)
        or not isinstance(contract, Mapping)
        or not isinstance(claims, Mapping)
        or not isinstance(rows, list)
        or len(rows) != 75
    ):
        raise CommonVoiceSileroV55SynthesisError(
            "Pre-QA literal text binding has an invalid contract."
        )
    _require_binding_input(inputs, "ready_manifest", base_manifest, rows=75)
    _require_binding_input(inputs, "silero_v5_5_model_lock", model_lock)
    _require_binding_input(inputs, "exact_route_audit", route_audit)
    archive_input = inputs.get("common_voice_archive")
    if (
        not isinstance(archive_input, Mapping)
        or archive_input.get("path") != str(archive)
        or archive_input.get("expected_size_bytes")
        != COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive_input.get("expected_sha256") != COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256
        or archive_input.get("identity_verified_before_metadata_read") is not True
    ):
        raise CommonVoiceSileroV55SynthesisError(
            "Text binding does not pin the expected Common Voice archive."
        )
    if (
        contract.get("ready_rows_only") is not True
        or contract.get("literal_source_text_only") is not True
        or contract.get("external_text_normalizer_or_stress_model") != "forbidden"
        or contract.get("text_replacement_or_reselection") != "forbidden"
        or contract.get("audio_or_duration_used_for_text_binding") is not False
        or contract.get("detector_or_metric_used") is not False
        or claims.get("audio_extraction_performed") is not False
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("acoustic_review_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
        or claims.get("future_synthesis_must_create_exactly_one_fixed_eugene_wav_per_bound_text")
        is not True
        or claims.get("failed_synthesis_or_qa_rows_must_not_be_replaced_or_backfilled")
        is not True
    ):
        raise CommonVoiceSileroV55SynthesisError(
            "Text binding does not retain the synthesis governance boundary."
        )
    bound: dict[str, Mapping[str, object]] = {}
    text_hashes: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise CommonVoiceSileroV55SynthesisError(f"Text binding row {index} is invalid.")
        sample_id = row.get("sample_id")
        text_id = row.get("text_id")
        text_hash = row.get("text_hash")
        if (
            not isinstance(sample_id, str)
            or not isinstance(text_id, str)
            or sample_id in bound
            or text_hash in text_hashes
            or _sha256(text_hash, f"Text binding row {index} text hash") != text_hash
            or row.get("source_split") != "test"
            or row.get("literal_text_sha256") != text_hash
            or row.get("normalizer_id") != "silero_v5_5_ru_literal_whitespace_only_v1"
            or row.get("normalized_text_sha256") != text_hash
        ):
            raise CommonVoiceSileroV55SynthesisError(
                f"Text binding row {index} violates the literal one-to-one contract."
            )
        bound[sample_id] = row
        text_hashes.add(text_hash)
    return bound


def require_route_audit(path: Path, model_lock: Path) -> None:
    """Check the completed audit's narrow exact-route claim without broadening it."""

    audit = _json_object(path, "Silero V5.5 exact-route audit")
    lock = audit.get("model_lock")
    runtime_policy = audit.get("runtime_policy")
    route_gate = audit.get("route_gate")
    claims = audit.get("claims")
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol_id") != ROUTE_AUDIT_PROTOCOL_ID
        or not isinstance(lock, Mapping)
        or lock.get("path") != model_lock.as_posix()
        or _sha256(lock.get("sha256"), "Route audit model lock SHA-256")
        != sha256_file(model_lock)
        or lock.get("model_id") != "silero_v5_5_ru_eugene"
        or not isinstance(runtime_policy, Mapping)
        or runtime_policy.get("fixed_voice_id") != SILERO_V5_5_FIXED_SPEAKER
        or runtime_policy.get("sample_rate") != SILERO_V5_5_SAMPLE_RATE
        or runtime_policy.get("reference_audio") != "forbidden"
        or runtime_policy.get("voice_cloning") is not False
        or runtime_policy.get("text_input_only") is not True
        or runtime_policy.get("ssml") != "forbidden"
        or runtime_policy.get("voice_path") != "forbidden"
        or runtime_policy.get("symbol_durs") != "forbidden"
        or not isinstance(route_gate, Mapping)
        or route_gate.get("novelty_claim") != "unseen_exact_generator_route"
        or route_gate.get("exact_route_overlap_rows") != 0
        or route_gate.get("architecture_independence_claim") is not False
        or route_gate.get("speaker_independence_claim") is not False
        or not isinstance(claims, Mapping)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("vendor_family_independence") is not False
        or claims.get("speaker_independence") is not False
        or claims.get("reference_audio_or_voice_cloning_used") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise CommonVoiceSileroV55SynthesisError("Silero V5.5 exact-route audit is invalid.")


def source_texts(
    archive: Path,
    base_rows: Sequence[ManifestRow],
    binding: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Re-read only pinned test TSV text and prove it remains literal and bound."""

    records = load_common_voice_metadata_from_archive(archive, ("test",))
    records_by_sample = {
        f"{COMMON_VOICE_RU_V24_SOURCE_ID}:{Path(record.clip_name).stem}": record
        for record in records
    }
    texts: dict[str, str] = {}
    for base in base_rows:
        record = records_by_sample.get(base.sample_id)
        bound = binding.get(base.sample_id)
        if record is None or bound is None:
            raise CommonVoiceSileroV55SynthesisError(
                f"Pinned archive or text binding lacks {base.sample_id!r}."
            )
        text_hash = hashlib.sha256(record.sentence.encode("utf-8")).hexdigest()
        literal_text = normalize_silero_v5_5_text(record.sentence)
        if (
            record.split != "test"
            or record.sentence_id != base.text_id
            or text_hash != base.text_hash
            or literal_text != record.sentence
            or bound.get("text_id") != base.text_id
            or bound.get("text_hash") != base.text_hash
        ):
            raise CommonVoiceSileroV55SynthesisError(
                f"Literal source text no longer matches {base.sample_id!r}."
            )
        texts[base.sample_id] = literal_text
    return texts


def _relative_to_data_root(data_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise CommonVoiceSileroV55SynthesisError(
            "Synthetic asset output must stay below data root."
        ) from error


def _validate_base_rows(rows: Sequence[ManifestRow]) -> None:
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
        raise CommonVoiceSileroV55SynthesisError(
            "Synthesis base must be exactly 75 unique Common Voice RU test bona-fide rows."
        )


def _write_one(
    *,
    output: Path,
    model: Any,
    text: str,
    runtime: SileroV55Runtime,
) -> tuple[str, float, int]:
    temp_output = output.with_name(f".{output.stem}.part.wav")
    if temp_output.exists() or output.exists():
        raise CommonVoiceSileroV55SynthesisError(
            "A supposedly new synthetic output already exists."
        )
    synthesize_silero_v5_5(model=model, text=text, runtime=runtime, output=temp_output)
    info = sf.info(temp_output)
    if (
        info.samplerate != runtime.sample_rate
        or info.channels != 1
        or info.frames <= 0
        or info.format != "WAV"
        or info.subtype != "PCM_16"
    ):
        raise CommonVoiceSileroV55SynthesisError(
            "Fixed-eugene synthesis did not produce a non-empty mono PCM-16 WAV at 48 kHz."
        )
    temp_output.replace(output)
    return sha256_file(output), float(info.duration), info.samplerate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--text-binding", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
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
            raise CommonVoiceSileroV55SynthesisError(
                "Synthesis output directory, manifest, and report must all be distinct and new."
            )
        data_root = arguments.data_root.resolve(strict=True)
        _relative_to_data_root(data_root, arguments.output_directory)
        arguments.output_directory.parent.mkdir(parents=True, exist_ok=True)
        base_rows = tuple(load_manifest(arguments.base_manifest))
        validate_manifest(base_rows)
        _validate_base_rows(base_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        require_valid_assets(base_rows, data_root)
        if SILERO_V5_5_SOURCE_ID not in ledger:
            raise LicenseLedgerError(["Silero V5.5 source is absent from the license ledger."])
        binding = require_text_binding(
            arguments.text_binding,
            base_manifest=arguments.base_manifest,
            archive=arguments.archive,
            model_lock=arguments.model_lock,
            route_audit=arguments.route_audit,
        )
        if set(binding) != {row.sample_id for row in base_rows}:
            raise CommonVoiceSileroV55SynthesisError(
                "Literal text binding does not cover exactly the ready base rows."
            )
        require_route_audit(arguments.route_audit, arguments.model_lock)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1 or lock.models[0].model_id != "silero_v5_5_ru_eugene":
            raise ResearchTtsError("Synthesis requires exactly the pinned Silero V5.5 eugene lock.")
        model_spec: ResearchTtsModel = lock.models[0]
        runtime = load_silero_v5_5_runtime(model_spec)
        verified = verify_research_tts_model_lock(arguments.model_root, lock)[model_spec.model_id]
        texts = source_texts(arguments.archive, base_rows, binding)
        model = load_silero_v5_5_model(
            verified[runtime.package_path], runtime, torch.device("cpu")
        )
        arguments.output_directory.mkdir(parents=False, exist_ok=False)
        rows: list[ManifestRow] = []
        generated: list[dict[str, object]] = []
        for index, base in enumerate(sorted(base_rows, key=lambda row: row.sample_id), start=1):
            file_key = hashlib.sha256(base.sample_id.encode("utf-8")).hexdigest()[:20]
            output = arguments.output_directory / f"{file_key}-{base.text_hash[:12]}.wav"
            audio_sha256, duration_s, original_sr = _write_one(
                output=output, model=model, text=texts[base.sample_id], runtime=runtime
            )
            row = silero_v5_5_spoof_row(
                base_row=base,
                model=model_spec,
                relative_path=_relative_to_data_root(data_root, output),
                sha256=audio_sha256,
                duration_s=duration_s,
                original_sr=original_sr,
                created_at=arguments.created_at,
                device=DEVICE_ID,
            )
            rows.append(row)
            generated.append(
                {
                    "base_sample_id": base.sample_id,
                    "spoof_sample_id": row.sample_id,
                    "text_id": row.text_id,
                    "text_hash": row.text_hash,
                    "relative_path": row.relative_path,
                    "audio_sha256": row.sha256,
                    "duration_s": row.duration_s,
                }
            )
            if index % 5 == 0 or index == len(base_rows):
                print(json.dumps({"status": "running", "generated_rows": index}), flush=True)
        if (
            len(rows) != 75
            or len({row.sample_id for row in rows}) != 75
            or len({row.text_id for row in rows}) != 75
            or len({row.text_hash for row in rows}) != 75
            or len({item["base_sample_id"] for item in generated}) != 75
        ):
            raise CommonVoiceSileroV55SynthesisError(
                "Synthesis did not create exactly one spoof row per bound text."
            )
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
        require_valid_assets(rows, data_root)
        with tempfile.TemporaryDirectory(
            prefix="kds-silero-v5-5-pre-qa-metadata-", dir=arguments.output_report.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged_manifest = stage / arguments.output_manifest.name
            staged_report = stage / arguments.output_report.name
            write_manifest(staged_manifest, rows)
            report = {
                "schema_version": 1,
                "protocol_id": SYNTHESIS_PROTOCOL_ID,
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
                "model_lock": {
                    "path": arguments.model_lock.as_posix(),
                    "sha256": sha256_file(arguments.model_lock),
                    "model_id": model_spec.model_id,
                    "fixed_speaker": runtime.fixed_speaker,
                    "sample_rate": runtime.sample_rate,
                    "verified_artifacts": {
                        name: sha256_file(model_path) for name, model_path in verified.items()
                    },
                },
                "output_manifest": {
                    "path": arguments.output_manifest.as_posix(),
                    "sha256": sha256_file(staged_manifest),
                    "rows": len(rows),
                },
                "generation_policy": {
                    "device": DEVICE_ID,
                    "exactly_one_synthetic_per_bound_base": True,
                    "fixed_profile": SILERO_V5_5_FIXED_SPEAKER,
                    "reference_audio_or_voice_cloning_used": False,
                    "post_selection_replacement_or_backfill": False,
                    "resynthesis_after_failure": "forbidden",
                    "partial_run_behavior": (
                        "raw output directory remains write-locked; no automatic retry or cleanup"
                    ),
                },
                "claims": {
                    "audio_extraction_performed": False,
                    "synthetic_audio_generated": True,
                    "technical_decode_qa_vad_performed": False,
                    "acoustic_review_performed": False,
                    "binary_pairing_performed": False,
                    "detector_inference_performed": False,
                    "detector_inference_authorized": False,
                },
                "generated": generated,
            }
            staged_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in outputs):
                raise CommonVoiceSileroV55SynthesisError(
                    "A synthesis metadata output appeared while staging."
                )
            staged_manifest.replace(arguments.output_manifest)
            staged_report.replace(arguments.output_report)
    except (
        CommonVoiceIngestionError,
        CommonVoiceSileroV55SynthesisError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        ResearchTtsError,
        SileroV55Error,
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
                "generated_rows": len(rows),
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
