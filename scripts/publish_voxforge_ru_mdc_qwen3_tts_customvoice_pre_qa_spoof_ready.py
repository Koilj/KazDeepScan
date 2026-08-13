"""Bind immutable Qwen raw synthesis and technical QA into retained VoxForge spoof rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    MANIFEST_FIELD_ORDER,
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
)
from kds.data.preprocess import processed_relative_path
from kds.data.qwen3_tts_customvoice_candidate import QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID

_SYNTHESIS_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-synthesis-v1"
_TECHNICAL_QA_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-technical-qa-v1"
_HEX = frozenset("0123456789abcdef")


class VoxForgeQwenSpoofReadyError(ValueError):
    """Raised when technical QA attempts to change a write-once Qwen synthesis layer."""


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgeQwenSpoofReadyError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise VoxForgeQwenSpoofReadyError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise VoxForgeQwenSpoofReadyError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def require_synthesis_receipt(path: Path, raw_manifest: Path) -> None:
    """Require the full completed 79-text synthesis before technical QA can publish."""

    receipt = _object(path, "Qwen pre-QA synthesis receipt")
    output = receipt.get("output_manifest")
    policy = receipt.get("generation_policy")
    claims = receipt.get("claims")
    generated = receipt.get("generated")
    failures = receipt.get("failed_attempts")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != _SYNTHESIS_PROTOCOL_ID
        or not isinstance(output, Mapping)
        or output.get("path") != raw_manifest.as_posix()
        or output.get("rows") != 79
        or _sha256(output.get("sha256"), "Synthesis raw manifest SHA-256")
        != sha256_file(raw_manifest)
        or not isinstance(policy, Mapping)
        or policy.get("exactly_one_synthetic_per_bound_base") is not True
        or policy.get("exactly_one_attempt_per_bound_text") is not True
        or policy.get("successful_one_to_one_synthetic_rows") != 79
        or policy.get("failed_attempt_rows") != 0
        or policy.get("post_selection_replacement_or_backfill") is not False
        or policy.get("resynthesis_after_failure") != "forbidden"
        or not isinstance(generated, list)
        or len(generated) != 79
        or not isinstance(failures, list)
        or failures
        or not isinstance(claims, Mapping)
        or claims.get("synthetic_audio_generated") is not True
        or claims.get("technical_decode_qa_vad_performed") is not False
        or claims.get("acoustic_review_performed") is not False
        or claims.get("binary_pairing_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise VoxForgeQwenSpoofReadyError(
            "Qwen synthesis receipt does not authorize this narrow technical QA step."
        )


def _validate_raw_rows(rows: Sequence[ManifestRow]) -> None:
    if (
        len(rows) != 79
        or len({row.sample_id for row in rows}) != 79
        or len({row.text_id for row in rows}) != 79
        or len({row.text_hash for row in rows}) != 79
        or any(
            row.split != "test"
            or row.label != "spoof"
            or row.language != "ru"
            or row.source_name != QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
            or row.voice_id != "qwen3_tts_customvoice:aiden"
            or row.original_sr != 24_000
            for row in rows
        )
    ):
        raise VoxForgeQwenSpoofReadyError(
            "Raw synthesis must retain exactly 79 unique fixed-Aiden Russian spoof rows."
        )


def rejection_accounting(
    *,
    raw_rows: Sequence[ManifestRow],
    ready_rows: Sequence[ManifestRow],
    report: Mapping[str, object],
    raw_manifest: Path,
) -> tuple[Mapping[str, object], ...]:
    """Require each raw synthesis row to be retained once or rejected once without replacement."""

    rejected = report.get("rejected_rows")
    if (
        report.get("input_manifest") != str(raw_manifest)
        or report.get("reused_rows") != 0
        or report.get("published_rows") != len(ready_rows)
        or not isinstance(rejected, list)
    ):
        raise VoxForgeQwenSpoofReadyError("Technical QA report has invalid count accounting.")
    raw_by_id = {row.sample_id: row for row in raw_rows}
    ready_ids = {row.sample_id for row in ready_rows}
    rejected_by_id: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(rejected, start=1):
        if not isinstance(item, Mapping):
            raise VoxForgeQwenSpoofReadyError(f"Technical QA rejection row {index} is invalid.")
        sample_id = item.get("sample_id")
        relative_path = item.get("relative_path")
        detail = item.get("detail")
        if not isinstance(sample_id, str):
            raise VoxForgeQwenSpoofReadyError(
                f"Technical QA rejection row {index} has no sample ID."
            )
        raw = raw_by_id.get(sample_id)
        if (
            raw is None
            or sample_id in rejected_by_id
            or sample_id in ready_ids
            or relative_path != raw.relative_path
            or not isinstance(detail, str)
            or not detail
        ):
            raise VoxForgeQwenSpoofReadyError(
                f"Technical QA rejection row {index} does not account for one raw asset."
            )
        rejected_by_id[sample_id] = item
    if set(raw_by_id) != ready_ids.union(rejected_by_id):
        raise VoxForgeQwenSpoofReadyError(
            "Technical QA does not account for every raw Qwen asset exactly once."
        )
    return tuple(rejected_by_id[sample_id] for sample_id in sorted(rejected_by_id))


def _validate_ready_rows(
    raw_rows: Sequence[ManifestRow], ready_rows: Sequence[ManifestRow]
) -> None:
    raw_by_id = {row.sample_id: row for row in raw_rows}
    if len({row.sample_id for row in ready_rows}) != len(ready_rows):
        raise VoxForgeQwenSpoofReadyError("Ready manifest repeats a Qwen spoof sample ID.")
    for ready in ready_rows:
        raw = raw_by_id.get(ready.sample_id)
        if raw is None:
            raise VoxForgeQwenSpoofReadyError(
                f"Ready manifest adds a spoof absent from raw synthesis: {ready.sample_id!r}."
            )
        for field in MANIFEST_FIELD_ORDER:
            if field in {"relative_path", "sha256", "duration_s", "codec"}:
                continue
            if getattr(ready, field) != getattr(raw, field):
                raise VoxForgeQwenSpoofReadyError(
                    f"Technical QA changes immutable {field} for {ready.sample_id!r}."
                )
        if ready.relative_path != processed_relative_path(raw) or ready.codec != "wav":
            raise VoxForgeQwenSpoofReadyError(
                f"Ready spoof path is not the normal processed path for {ready.sample_id!r}."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--synthesis-receipt", type=Path, required=True)
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
            raise VoxForgeQwenSpoofReadyError(
                "Technical-QA receipt output must be new with an existing parent."
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
        require_synthesis_receipt(arguments.synthesis_receipt, arguments.raw_manifest)
        rejection_rows = rejection_accounting(
            raw_rows=raw_rows,
            ready_rows=ready_rows,
            report=_object(arguments.rejection_report, "technical QA rejection report"),
            raw_manifest=arguments.raw_manifest,
        )
        receipt = {
            "schema_version": 1,
            "protocol_id": _TECHNICAL_QA_PROTOCOL_ID,
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
                    "AudioPreparationPipeline: decode, mono PCM WAV 16 kHz, quality checks, "
                    "WebRTC VAD"
                ),
                "raw_rows": len(raw_rows),
                "ready_rows": len(ready_rows),
                "rejected_rows": list(rejection_rows),
                "reused_rows": 0,
                "resynthesis_or_replacement_or_backfill": False,
            },
            "claims": {
                "synthetic_audio_generated": True,
                "technical_decode_qa_vad_performed": True,
                "acoustic_review_performed": False,
                "binary_pairing_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "future_pairing_must_use_only_retained_ready_spoof_rows": True,
            },
        }
        arguments.output.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        LicenseLedgerError,
        ManifestError,
        OSError,
        ValueError,
        VoxForgeQwenSpoofReadyError,
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
                "ready_rows": len(ready_rows),
                "rejected_rows": len(rejection_rows),
                "receipt": str(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
