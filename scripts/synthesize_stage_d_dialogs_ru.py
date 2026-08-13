"""Create exactly one Dialogs-RU spoof WAV for each frozen Stage-D Common Voice row."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.common_voice import load_common_voice_metadata_from_archive
from kds.data.dialogs_ru_vits2 import DialogsRuVits2Error, load_dialogs_ru_vits2
from kds.data.dialogs_ru_vits2_candidate import (
    DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID,
    DialogsRuVits2CandidateError,
    dialogs_ru_vits2_spoof_row,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock


class StageDDialogsSynthesisError(ValueError):
    """Raised when the immutable Stage-D synthesis contract is not satisfied."""


def _load_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageDDialogsSynthesisError(f"Cannot read {label}: {error}") from error
    if not isinstance(raw, dict):
        raise StageDDialogsSynthesisError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], raw)


def _require_text_binding(
    path: Path,
    *,
    candidate_manifest: Path,
    model_lock: Path,
    archive: Path,
) -> dict[str, Mapping[str, object]]:
    binding = _load_object(path, "Stage-D literal text binding")
    candidate = binding.get("candidate_manifest")
    model = binding.get("model_lock")
    source = binding.get("common_voice_archive")
    rows = binding.get("rows")
    contract = binding.get("input_contract")
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol_id") != "stage-d-common-voice-ru-literal-text-binding-v1"
        or not isinstance(candidate, dict)
        or candidate.get("path") != candidate_manifest.as_posix()
        or candidate.get("sha256") != sha256_file(candidate_manifest)
        or candidate.get("rows") != 73
        or not isinstance(model, dict)
        or model.get("path") != model_lock.as_posix()
        or model.get("sha256") != sha256_file(model_lock)
        or not isinstance(source, dict)
        or source.get("path") != str(archive)
        or source.get("size_bytes") != archive.stat().st_size
        or source.get("sha256") != sha256_file(archive)
        or not isinstance(contract, dict)
        or contract.get("synthesis_uses_literal_source_text") is not True
        or contract.get("text_replacement_or_reselection") != "forbidden"
        or contract.get("detector_or_metric_used") is not False
        or not isinstance(rows, list)
        or len(rows) != 73
    ):
        raise StageDDialogsSynthesisError("Stage-D literal text binding is invalid or changed.")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise StageDDialogsSynthesisError("Stage-D literal text binding row is invalid.")
        sample_id = row.get("sample_id")
        text_hash = row.get("text_hash")
        if (
            not isinstance(sample_id, str)
            or sample_id in result
            or not isinstance(text_hash, str)
            or len(text_hash) != 64
        ):
            raise StageDDialogsSynthesisError(
                "Stage-D literal text binding repeats or corrupts a row."
            )
        result[sample_id] = cast(Mapping[str, object], row)
    return result


def _require_route_audit(path: Path, model_lock: Path) -> None:
    audit = _load_object(path, "Stage-D Dialogs-RU route audit")
    lock = audit.get("model_lock")
    route_gate = audit.get("route_gate")
    claims = audit.get("claims")
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol_id") != "stage-d-dialogs-ru-vits2-exact-route-audit-v1"
        or not isinstance(lock, dict)
        or lock.get("path") != model_lock.as_posix()
        or lock.get("sha256") != sha256_file(model_lock)
        or not isinstance(route_gate, dict)
        or route_gate.get("exact_route_overlap_rows") != 0
        or not isinstance(claims, dict)
        or claims.get("exact_generator_route_absent_from_historical_manifests") is not True
        or claims.get("architecture_family_novelty") is not False
        or claims.get("speaker_independence") is not False
    ):
        raise StageDDialogsSynthesisError("Stage-D Dialogs-RU exact-route audit is invalid.")


def _source_texts(archive: Path, base_rows: list[ManifestRow]) -> dict[str, str]:
    records = load_common_voice_metadata_from_archive(archive, ["train", "dev", "test"])
    records_by_id = {
        f"common_voice_ru_v24:{Path(record.clip_name).stem}": record for record in records
    }
    texts: dict[str, str] = {}
    for row in base_rows:
        record = records_by_id.get(row.sample_id)
        if record is None:
            raise StageDDialogsSynthesisError(
                f"Common Voice archive lacks frozen row {row.sample_id!r}."
            )
        actual_hash = hashlib.sha256(record.sentence.encode("utf-8")).hexdigest()
        if actual_hash != row.text_hash:
            raise StageDDialogsSynthesisError(
                f"Common Voice source text changes frozen hash for {row.sample_id!r}."
            )
        texts[row.sample_id] = record.sentence
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--text-binding", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--common-voice-archive", type=Path, required=True)
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
        outputs = (arguments.output_manifest, arguments.output_report)
        if (
            arguments.output_directory.exists()
            or any(path.exists() or not path.parent.is_dir() for path in outputs)
            or len(set(outputs)) != len(outputs)
        ):
            raise StageDDialogsSynthesisError("Stage-D synthesis outputs must be distinct and new.")
        data_root = arguments.data_root.resolve(strict=True)
        output_directory = arguments.output_directory.resolve(strict=False)
        if not output_directory.is_relative_to(data_root):
            raise StageDDialogsSynthesisError("Stage-D asset output must remain below data-root.")
        base_rows = load_manifest(arguments.candidate_manifest)
        validate_manifest(base_rows)
        if (
            len(base_rows) != 73
            or any(
                row.split != "test"
                or row.label != "bonafide"
                or row.language != "ru"
                or row.source_name != "common_voice_ru_v24"
                for row in base_rows
            )
        ):
            raise StageDDialogsSynthesisError(
                "Stage-D base must remain 73 frozen Common Voice RU rows."
            )
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        require_valid_assets(base_rows, data_root)
        if DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID not in ledger:
            raise LicenseLedgerError(["Stage-D synthetic source is absent from license ledger."])
        binding_rows = _require_text_binding(
            arguments.text_binding,
            candidate_manifest=arguments.candidate_manifest,
            model_lock=arguments.model_lock,
            archive=arguments.common_voice_archive,
        )
        if set(binding_rows) != {row.sample_id for row in base_rows}:
            raise StageDDialogsSynthesisError(
                "Text binding does not exactly cover frozen Stage-D rows."
            )
        _require_route_audit(arguments.route_audit, arguments.model_lock)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Stage-D synthesis requires exactly one Dialogs-RU model.")
        model = lock.models[0]
        texts = _source_texts(arguments.common_voice_archive, base_rows)
        runtime = load_dialogs_ru_vits2(arguments.model_root, model)
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        stage_assets = Path(
            tempfile.mkdtemp(prefix=".kds-stage-d-dialogs-assets-", dir=output_directory.parent)
        )
        stage_metadata = Path(
            tempfile.mkdtemp(
                prefix=".kds-stage-d-dialogs-metadata-",
                dir=arguments.output_report.parent,
            )
        )
        try:
            rows: list[ManifestRow] = []
            generated: list[dict[str, object]] = []
            for index, base in enumerate(sorted(base_rows, key=lambda row: row.sample_id), start=1):
                source_text = texts[base.sample_id]
                if binding_rows[base.sample_id].get("text_hash") != base.text_hash:
                    raise StageDDialogsSynthesisError("Text binding changes a frozen text hash.")
                prepared = runtime.prepare_text(source_text)
                waveform = runtime.synthesize(prepared)
                file_key = hashlib.sha256(base.sample_id.encode("utf-8")).hexdigest()[:20]
                staged_audio = stage_assets / f"{file_key}-{base.text_hash[:12]}.wav"
                sf.write(staged_audio, waveform, runtime.sample_rate, subtype="PCM_16")
                info = sf.info(staged_audio)
                if info.samplerate != runtime.sample_rate or info.frames <= 0:
                    raise StageDDialogsSynthesisError(
                        "Stage-D raw WAV failed immediate decode check."
                    )
                final_audio = output_directory / staged_audio.name
                row = dialogs_ru_vits2_spoof_row(
                    base_row=base,
                    model=model,
                    runtime=runtime,
                    prepared=prepared,
                    relative_path=final_audio.relative_to(data_root).as_posix(),
                    sha256=sha256_file(staged_audio),
                    duration_s=float(info.duration),
                    created_at=arguments.created_at,
                )
                rows.append(row)
                generated.append(
                    {
                        "base_sample_id": base.sample_id,
                        "spoof_sample_id": row.sample_id,
                        "text_hash": base.text_hash,
                        "audio_sha256": row.sha256,
                        "duration_s": row.duration_s,
                        "tokenizer_dropped_characters": list(prepared.dropped_characters),
                    }
                )
                if index % 10 == 0 or index == len(base_rows):
                    print(
                        json.dumps(
                            {"status": "running", "generated_rows": index}, sort_keys=True
                        ),
                        flush=True,
                    )
            if len(rows) != 73 or len({row.text_id for row in rows}) != 73:
                raise StageDDialogsSynthesisError(
                    "Stage-D synthesis did not create exactly 73 one-to-one rows."
                )
            validate_manifest(rows)
            staged_manifest = stage_metadata / arguments.output_manifest.name
            write_manifest(staged_manifest, rows)
            report = {
                "schema_version": 1,
                "protocol_id": "stage-d-dialogs-ru-masha-neutral-synthesis-v1",
                "created_at": arguments.created_at,
                "candidate_manifest": {
                    "path": arguments.candidate_manifest.as_posix(),
                    "sha256": sha256_file(arguments.candidate_manifest),
                    "rows": len(base_rows),
                },
                "text_binding": {
                    "path": arguments.text_binding.as_posix(),
                    "sha256": sha256_file(arguments.text_binding),
                },
                "route_audit": {
                    "path": arguments.route_audit.as_posix(),
                    "sha256": sha256_file(arguments.route_audit),
                },
                "model_lock": {
                    "path": arguments.model_lock.as_posix(),
                    "sha256": sha256_file(arguments.model_lock),
                },
                "output_manifest": {
                    "path": arguments.output_manifest.as_posix(),
                    "sha256": sha256_file(staged_manifest),
                    "rows": len(rows),
                },
                "generated_rows": len(rows),
                "exactly_one_synthetic_per_frozen_base": True,
                "post_selection_backfill": False,
                "reference_audio_or_voice_cloning_used": False,
                "detector_inference_performed": False,
                "full_asset_acoustic_gate_passed": False,
                "generated": generated,
            }
            staged_report = stage_metadata / arguments.output_report.name
            staged_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if arguments.output_directory.exists() or any(path.exists() for path in outputs):
                raise StageDDialogsSynthesisError("A Stage-D output appeared while staging.")
            stage_assets.replace(output_directory)
            staged_manifest.replace(arguments.output_manifest)
            staged_report.replace(arguments.output_report)
        finally:
            shutil.rmtree(stage_assets, ignore_errors=True)
            shutil.rmtree(stage_metadata, ignore_errors=True)
    except (
        DialogsRuVits2CandidateError,
        DialogsRuVits2Error,
        LicenseLedgerError,
        ManifestError,
        ResearchTtsError,
        StageDDialogsSynthesisError,
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
