"""Bind write-once VoxCPM2 synthesis and technical QA into retained Denis spoof rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.denis_voxcpm2_candidate import (
    DENIS_VOXCPM2_SOURCE_ID,
    DENIS_VOXCPM2_SYNTHESIS_PROTOCOL_ID,
    DENIS_VOXCPM2_TECHNICAL_QA_PROTOCOL_ID,
    DENIS_VOXCPM2_TEXT_BINDING_PROTOCOL_ID,
    DENIS_VOXCPM2_VOICE_ID,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    MANIFEST_FIELD_ORDER,
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
)
from kds.data.preprocess import processed_relative_path

_BOUND_ROWS = 64
_MINIMUM_ROWS = 60
_HEX = frozenset("0123456789abcdef")


class DenisVoxCPM2SpoofReadyError(ValueError):
    """Raised when technical QA changes the write-once VoxCPM2 synthesis layer."""


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DenisVoxCPM2SpoofReadyError(f"Cannot read {label}: {path}.") from error
    if not isinstance(payload, dict):
        raise DenisVoxCPM2SpoofReadyError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise DenisVoxCPM2SpoofReadyError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def require_binding_outputs(
    path: Path,
    *,
    raw_manifest: Path,
    ready_manifest: Path,
    rejection_report: Path,
    technical_qa_receipt: Path,
) -> None:
    """Require the exact committed QA programs and output paths from the pre-synthesis gate."""

    binding = _object(path, "Denis VoxCPM2 text binding")
    programs = binding.get("frozen_programs")
    outputs = binding.get("output_contract")
    claims = binding.get("claims")
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol_id") != DENIS_VOXCPM2_TEXT_BINDING_PROTOCOL_ID
        or not isinstance(programs, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(claims, Mapping)
        or outputs.get("raw_manifest") != raw_manifest.as_posix()
        or outputs.get("ready_spoof_manifest") != ready_manifest.as_posix()
        or outputs.get("technical_qa_rejection_report") != rejection_report.as_posix()
        or outputs.get("technical_qa_receipt") != technical_qa_receipt.as_posix()
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise DenisVoxCPM2SpoofReadyError("Text binding does not authorize this QA output.")
    publisher = programs.get("technical_qa_publisher")
    preprocess = programs.get("preprocess_runner")
    publisher_path = Path(__file__).resolve()
    project_root = publisher_path.parent.parent
    publisher_project_path = publisher_path.relative_to(project_root).as_posix()
    preprocess_path = project_root / "scripts/preprocess_manifest.py"
    if (
        not isinstance(publisher, Mapping)
        or publisher.get("path") != publisher_project_path
        or _sha256(publisher.get("sha256"), "QA publisher SHA-256") != sha256_file(publisher_path)
        or not isinstance(preprocess, Mapping)
        or preprocess.get("path") != "scripts/preprocess_manifest.py"
        or _sha256(preprocess.get("sha256"), "Preprocess runner SHA-256")
        != sha256_file(preprocess_path)
    ):
        raise DenisVoxCPM2SpoofReadyError("Committed preprocessing or QA publisher changed.")


def require_synthesis_receipt(
    path: Path, *, raw_manifest: Path, text_binding: Path, raw_rows: int
) -> None:
    """Require all 64 one-shot calls to be accounted for before technical QA."""

    receipt = _object(path, "Denis VoxCPM2 synthesis receipt")
    binding = receipt.get("text_binding")
    output = receipt.get("output_manifest")
    policy = receipt.get("generation_policy")
    claims = receipt.get("claims")
    generated = receipt.get("generated")
    failures = receipt.get("failed_attempts")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != DENIS_VOXCPM2_SYNTHESIS_PROTOCOL_ID
        or not isinstance(binding, Mapping)
        or binding.get("path") != text_binding.as_posix()
        or binding.get("rows") != _BOUND_ROWS
        or _sha256(binding.get("sha256"), "Text-binding SHA-256") != sha256_file(text_binding)
        or not isinstance(output, Mapping)
        or output.get("path") != raw_manifest.as_posix()
        or output.get("rows") != raw_rows
        or _sha256(output.get("sha256"), "Raw manifest SHA-256") != sha256_file(raw_manifest)
        or not isinstance(policy, Mapping)
        or policy.get("bound_rows") != _BOUND_ROWS
        or policy.get("attempted_rows") != _BOUND_ROWS
        or policy.get("successful_rows") != raw_rows
        or policy.get("failed_attempt_rows") != _BOUND_ROWS - raw_rows
        or policy.get("exactly_one_attempt_per_bound_text") is not True
        or policy.get("reference_or_prompt_audio_used") is not False
        or policy.get("voice_cloning_or_lora_used") is not False
        or policy.get("semantic_normalizer_used") is not False
        or policy.get("denoiser_used") is not False
        or policy.get("retry_or_resynthesis_used") is not False
        or policy.get("post_selection_replacement_or_backfill") is not False
        or policy.get("resynthesis_after_failure") != "forbidden"
        or not isinstance(generated, list)
        or len(generated) != raw_rows
        or not isinstance(failures, list)
        or len(failures) != _BOUND_ROWS - raw_rows
        or not isinstance(claims, Mapping)
        or claims.get("synthetic_audio_generated") is not True
        or claims.get("technical_decode_qa_vad_performed") is not False
        or claims.get("acoustic_review_performed") is not False
        or claims.get("binary_pairing_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise DenisVoxCPM2SpoofReadyError(
            "Synthesis receipt does not authorize this narrow technical QA step."
        )


def _validate_raw_rows(rows: Sequence[ManifestRow]) -> None:
    if (
        not rows
        or len(rows) > _BOUND_ROWS
        or len({row.sample_id for row in rows}) != len(rows)
        or len({row.text_id for row in rows}) != len(rows)
        or len({row.text_hash for row in rows}) != len(rows)
        or any(
            row.split != "ood"
            or row.label != "spoof"
            or row.language != "ru"
            or row.source_name != DENIS_VOXCPM2_SOURCE_ID
            or row.voice_id != DENIS_VOXCPM2_VOICE_ID
            or row.original_sr != 48_000
            for row in rows
        )
    ):
        raise DenisVoxCPM2SpoofReadyError("Raw synthesis rows violate the frozen route.")


def _validate_ready_rows(
    raw_rows: Sequence[ManifestRow], ready_rows: Sequence[ManifestRow]
) -> None:
    raw_by_id = {row.sample_id: row for row in raw_rows}
    if len({row.sample_id for row in ready_rows}) != len(ready_rows):
        raise DenisVoxCPM2SpoofReadyError("Ready manifest repeats a spoof sample ID.")
    for ready in ready_rows:
        raw = raw_by_id.get(ready.sample_id)
        if raw is None:
            raise DenisVoxCPM2SpoofReadyError(
                f"Ready manifest adds a spoof absent from synthesis: {ready.sample_id!r}."
            )
        for field in MANIFEST_FIELD_ORDER:
            if field in {"relative_path", "sha256", "duration_s", "codec"}:
                continue
            if getattr(ready, field) != getattr(raw, field):
                raise DenisVoxCPM2SpoofReadyError(
                    f"Technical QA changes immutable {field} for {ready.sample_id!r}."
                )
        if ready.relative_path != processed_relative_path(raw) or ready.codec != "wav":
            raise DenisVoxCPM2SpoofReadyError(
                f"Ready spoof path is not the normal processed path: {ready.sample_id!r}."
            )


def rejection_accounting(
    *,
    raw_rows: Sequence[ManifestRow],
    ready_rows: Sequence[ManifestRow],
    report: Mapping[str, object],
    raw_manifest: Path,
) -> tuple[Mapping[str, object], ...]:
    """Account for every raw row exactly once as retained or rejected, with no reuse."""

    rejected = report.get("rejected_rows")
    if (
        report.get("input_manifest") != str(raw_manifest)
        or report.get("reused_rows") != 0
        or report.get("published_rows") != len(ready_rows)
        or not isinstance(rejected, list)
    ):
        raise DenisVoxCPM2SpoofReadyError("Technical-QA report count accounting is invalid.")
    raw_by_id = {row.sample_id: row for row in raw_rows}
    ready_ids = {row.sample_id for row in ready_rows}
    rejected_by_id: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(rejected, start=1):
        if not isinstance(item, Mapping):
            raise DenisVoxCPM2SpoofReadyError(f"QA rejection row {index} is invalid.")
        sample_id = item.get("sample_id")
        raw = raw_by_id.get(sample_id) if isinstance(sample_id, str) else None
        if (
            raw is None
            or sample_id in rejected_by_id
            or sample_id in ready_ids
            or item.get("relative_path") != raw.relative_path
            or not isinstance(item.get("detail"), str)
            or not item.get("detail")
        ):
            raise DenisVoxCPM2SpoofReadyError(
                f"QA rejection row {index} does not account for one raw asset."
            )
        rejected_by_id[cast(str, sample_id)] = item
    if set(raw_by_id) != ready_ids.union(rejected_by_id):
        raise DenisVoxCPM2SpoofReadyError(
            "Technical QA does not account for every synthesized asset exactly once."
        )
    return tuple(rejected_by_id[key] for key in sorted(rejected_by_id))


def _reason(detail: object) -> str:
    value = str(detail)
    for reason in ("insufficient_speech", "signal_too_quiet", "excessive_clipping"):
        if reason in value:
            return reason
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--synthesis-receipt", type=Path, required=True)
    parser.add_argument("--text-binding", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--rejection-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--published-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.published_at.replace("Z", "+00:00"))
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise DenisVoxCPM2SpoofReadyError(
                "Technical-QA receipt must be new with an existing parent."
            )
        require_binding_outputs(
            arguments.text_binding,
            raw_manifest=arguments.raw_manifest,
            ready_manifest=arguments.ready_manifest,
            rejection_report=arguments.rejection_report,
            technical_qa_receipt=arguments.output,
        )
        raw_rows = tuple(load_manifest(arguments.raw_manifest))
        ready_rows = tuple(load_manifest(arguments.ready_manifest))
        validate_manifest(raw_rows)
        validate_manifest(ready_rows)
        _validate_raw_rows(raw_rows)
        _validate_ready_rows(raw_rows, ready_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(raw_rows, ledger)
        validate_manifest_licenses(ready_rows, ledger)
        data_root = arguments.data_root.resolve(strict=True)
        require_valid_assets(raw_rows, data_root)
        require_valid_assets(ready_rows, data_root)
        require_synthesis_receipt(
            arguments.synthesis_receipt,
            raw_manifest=arguments.raw_manifest,
            text_binding=arguments.text_binding,
            raw_rows=len(raw_rows),
        )
        rejection_rows = rejection_accounting(
            raw_rows=raw_rows,
            ready_rows=ready_rows,
            report=_object(arguments.rejection_report, "technical-QA rejection report"),
            raw_manifest=arguments.raw_manifest,
        )
        outcome = (
            "target_64_met"
            if len(ready_rows) == _BOUND_ROWS
            else "minimum_60_met_but_target_not_met"
            if len(ready_rows) >= _MINIMUM_ROWS
            else "stop_below_minimum_60"
        )
        receipt = {
            "schema_version": 1,
            "protocol_id": DENIS_VOXCPM2_TECHNICAL_QA_PROTOCOL_ID,
            "published_at": arguments.published_at,
            "inputs": {
                "raw_manifest": {
                    "path": arguments.raw_manifest.as_posix(),
                    "sha256": sha256_file(arguments.raw_manifest),
                    "rows": len(raw_rows),
                },
                "synthesis_receipt": {
                    "path": arguments.synthesis_receipt.as_posix(),
                    "sha256": sha256_file(arguments.synthesis_receipt),
                },
                "text_binding": {
                    "path": arguments.text_binding.as_posix(),
                    "sha256": sha256_file(arguments.text_binding),
                    "rows": _BOUND_ROWS,
                },
                "rejection_report": {
                    "path": arguments.rejection_report.as_posix(),
                    "sha256": sha256_file(arguments.rejection_report),
                },
            },
            "outputs": {
                "ready_manifest": {
                    "path": arguments.ready_manifest.as_posix(),
                    "sha256": sha256_file(arguments.ready_manifest),
                    "rows": len(ready_rows),
                }
            },
            "technical_qa": {
                "pipeline": (
                    "AudioPreparationPipeline: decode, mono PCM-16 WAV 16 kHz, "
                    "RMS/clipping/DC measurement, WebRTC VAD"
                ),
                "bound_rows": _BOUND_ROWS,
                "synthesis_raw_rows": len(raw_rows),
                "ready_rows": len(ready_rows),
                "rejected_rows": len(rejection_rows),
                "rejection_reason_counts": dict(
                    sorted(Counter(_reason(row.get("detail")) for row in rejection_rows).items())
                ),
                "rejections": list(rejection_rows),
                "reused_rows": 0,
                "resynthesis_replacement_or_backfill": False,
            },
            "target_outcome": {
                "minimum_ready_pairs": _MINIMUM_ROWS,
                "target_ready_pairs": _BOUND_ROWS,
                "actual_ready_pairs": len(ready_rows),
                "status": outcome,
            },
            "claims": {
                "synthetic_audio_generated": True,
                "technical_decode_qa_vad_performed": True,
                "acoustic_review_performed": False,
                "binary_pairing_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "future_pairing_must_use_only_retained_ready_spoof_rows": True,
                "external_source_and_generator_family_holdout": True,
                "training_data_overlap_unverified": True,
                "single_bonafide_speaker": True,
                "speaker_independent": False,
                "speaker_robust": False,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
    except (LicenseLedgerError, ManifestError, OSError, ValueError) as error:
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
                "ready_rows": len(ready_rows),
                "rejected_rows": len(rejection_rows),
                "outcome": outcome,
                "receipt": str(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
