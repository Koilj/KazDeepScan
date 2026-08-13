"""Publish immutable 42-pair Common Voice/Silero V5.5 candidate after technical QA."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.silero_v5_5 import SILERO_V5_5_SOURCE_ID

TECHNICAL_QA_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-spoof-technical-qa-v1"
PAIRING_PROTOCOL_ID = "common-voice-ru-v24-silero-v5-5-eugene-pre-qa-pairing-v1"
_HEX = frozenset("0123456789abcdef")


class CommonVoiceSileroV55PairingError(ValueError):
    """Raised when an immutable pre-QA pair would be incomplete or substituted."""


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CommonVoiceSileroV55PairingError(f"Cannot read {label}: {error}") from error
    if not isinstance(raw, dict):
        raise CommonVoiceSileroV55PairingError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], raw)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise CommonVoiceSileroV55PairingError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_pinned_file(
    values: Mapping[str, object], name: str, expected_path: Path, *, rows: int | None = None
) -> None:
    value = values.get(name)
    if (
        not isinstance(value, Mapping)
        or value.get("path") != expected_path.as_posix()
        or _sha256(value.get("sha256"), f"Technical QA {name} SHA-256")
        != sha256_file(expected_path)
        or (rows is not None and value.get("rows") != rows)
    ):
        raise CommonVoiceSileroV55PairingError(
            f"Technical QA receipt has an invalid {name} binding."
        )


def require_technical_qa_receipt(
    path: Path, *, raw_manifest: Path, ready_manifest: Path
) -> set[str]:
    """Return exactly the 33 raw sample IDs permanently rejected by technical QA."""

    receipt = _object(path, "spoof technical QA receipt")
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    qa = receipt.get("technical_qa")
    claims = receipt.get("claims")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != TECHNICAL_QA_PROTOCOL_ID
        or not isinstance(inputs, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(qa, Mapping)
        or not isinstance(claims, Mapping)
    ):
        raise CommonVoiceSileroV55PairingError(
            "Spoof technical QA receipt has an invalid contract."
        )
    _require_pinned_file(inputs, "raw_manifest", raw_manifest, rows=75)
    _require_pinned_file(outputs, "ready_manifest", ready_manifest, rows=42)
    rejected = qa.get("rejected_rows")
    if (
        qa.get("raw_rows") != 75
        or qa.get("ready_rows") != 42
        or qa.get("reused_rows") != 0
        or qa.get("resynthesis_or_replacement_or_backfill") is not False
        or not isinstance(rejected, list)
        or len(rejected) != 33
        or claims.get("synthetic_audio_generated") is not True
        or claims.get("technical_decode_qa_vad_performed") is not True
        or claims.get("acoustic_review_performed") is not False
        or claims.get("binary_pairing_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
        or claims.get("future_pairing_must_use_only_retained_ready_spoof_rows") is not True
    ):
        raise CommonVoiceSileroV55PairingError(
            "Spoof technical QA receipt violates the immutable candidate boundary."
        )
    rejected_ids: set[str] = set()
    for index, item in enumerate(rejected, start=1):
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("sample_id"), str)
            or item["sample_id"] in rejected_ids
            or not isinstance(item.get("relative_path"), str)
            or not isinstance(item.get("detail"), str)
            or "insufficient_speech" not in item["detail"]
        ):
            raise CommonVoiceSileroV55PairingError(
                f"Spoof technical QA rejection {index} is invalid."
            )
        rejected_ids.add(item["sample_id"])
    return rejected_ids


def _rows_by_text(rows: Sequence[ManifestRow], label: str) -> dict[str, ManifestRow]:
    result = {row.text_id: row for row in rows}
    if len(result) != len(rows):
        raise CommonVoiceSileroV55PairingError(f"{label} repeats a text ID.")
    return result


def _validate_roles(
    base: Sequence[ManifestRow], raw: Sequence[ManifestRow], ready: Sequence[ManifestRow]
) -> None:
    if len(base) != 75 or any(
        row.split != "test"
        or row.label != "bonafide"
        or row.language != "ru"
        or row.source_name != "common_voice_ru_v24"
        or row.codec != "wav"
        for row in base
    ):
        raise CommonVoiceSileroV55PairingError(
            "Base must remain the exact 75-row Common Voice ready layer."
        )
    if len(raw) != 75 or len(ready) != 42 or any(
        row.split != "test"
        or row.label != "spoof"
        or row.language != "ru"
        or row.source_name != SILERO_V5_5_SOURCE_ID
        or row.codec != "wav"
        for row in (*raw, *ready)
    ):
        raise CommonVoiceSileroV55PairingError(
            "Spoof raw/ready layers do not retain the locked 75/42 Silero route."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--raw-spoof-manifest", type=Path, required=True)
    parser.add_argument("--ready-spoof-manifest", type=Path, required=True)
    parser.add_argument("--technical-qa-receipt", type=Path, required=True)
    parser.add_argument("--synthesis-receipt", type=Path, required=True)
    parser.add_argument("--text-binding", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-candidate", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = (arguments.output_candidate, arguments.output_receipt)
    try:
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if len(set(outputs)) != len(outputs) or any(
            path.exists() or not path.parent.is_dir() for path in outputs
        ):
            raise CommonVoiceSileroV55PairingError(
                "Pairing outputs must be distinct, new and have existing parents."
            )
        base = tuple(load_manifest(arguments.base_manifest))
        raw = tuple(load_manifest(arguments.raw_spoof_manifest))
        ready = tuple(load_manifest(arguments.ready_spoof_manifest))
        for rows in (base, raw, ready):
            validate_manifest(rows)
        _validate_roles(base, raw, ready)
        ledger = load_license_ledger(arguments.license_ledger)
        data_root = arguments.data_root.resolve(strict=True)
        for rows in (base, raw, ready):
            validate_manifest_licenses(rows, ledger)
            require_valid_assets(rows, data_root)
        rejected = require_technical_qa_receipt(
            arguments.technical_qa_receipt,
            raw_manifest=arguments.raw_spoof_manifest,
            ready_manifest=arguments.ready_spoof_manifest,
        )
        base_by_text = _rows_by_text(base, "Base manifest")
        raw_by_text = _rows_by_text(raw, "Raw spoof manifest")
        ready_by_text = _rows_by_text(ready, "Ready spoof manifest")
        if set(raw_by_text) != set(base_by_text):
            raise CommonVoiceSileroV55PairingError(
                "Raw spoof layer does not exactly cover every frozen base text."
            )
        raw_by_id = {row.sample_id: row for row in raw}
        if not rejected.issubset(raw_by_id):
            raise CommonVoiceSileroV55PairingError(
                "Technical QA receipt rejects a spoof sample absent from raw synthesis."
            )
        expected_texts = {row.text_id for row in raw if row.sample_id not in rejected}
        if set(ready_by_text) != expected_texts:
            raise CommonVoiceSileroV55PairingError(
                "Ready spoof layer is not raw synthesis minus exactly the technical QA rejects."
            )
        paired_base = tuple(base_by_text[text_id] for text_id in sorted(expected_texts))
        if any(
            base_row.text_hash != ready_by_text[base_row.text_id].text_hash
            for base_row in paired_base
        ):
            raise CommonVoiceSileroV55PairingError(
                "A retained spoof row changes its paired Common Voice text hash."
            )
        candidate = tuple(
            sorted(
                (*paired_base, *(ready_by_text[row.text_id] for row in paired_base)),
                key=lambda row: (row.text_id, row.label),
            )
        )
        validate_manifest(candidate)
        with tempfile.TemporaryDirectory(
            prefix="kds-silero-v5-5-pre-qa-pairs-", dir=arguments.output_receipt.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged_candidate = stage / arguments.output_candidate.name
            staged_receipt = stage / arguments.output_receipt.name
            write_manifest(staged_candidate, candidate)
            receipt = {
                "schema_version": 1,
                "protocol_id": PAIRING_PROTOCOL_ID,
                "created_at": arguments.created_at,
                "inputs": {
                    name: {"path": path.as_posix(), "sha256": sha256_file(path)}
                    for name, path in {
                        "base_manifest": arguments.base_manifest,
                        "raw_spoof_manifest": arguments.raw_spoof_manifest,
                        "ready_spoof_manifest": arguments.ready_spoof_manifest,
                        "technical_qa_receipt": arguments.technical_qa_receipt,
                        "synthesis_receipt": arguments.synthesis_receipt,
                        "text_binding": arguments.text_binding,
                        "route_audit": arguments.route_audit,
                        "license_ledger": arguments.license_ledger,
                    }.items()
                },
                "counts": {
                    "frozen_ready_bona_fide_rows": len(base),
                    "raw_spoof_rows": len(raw),
                    "technical_qa_rejected_spoof_rows": len(rejected),
                    "ready_spoof_rows": len(ready),
                    "retained_pairs": len(paired_base),
                    "candidate_assets": len(candidate),
                },
                "output_candidate": {
                    "path": arguments.output_candidate.as_posix(),
                    "sha256": sha256_file(staged_candidate),
                    "rows": len(candidate),
                },
                "decision_rule": {
                    "qa_reject_excludes_entire_pair": True,
                    "post_selection_backfill": False,
                    "resynthesis_after_qa": False,
                    "metric_or_detector_based_selection": False,
                    "detector_inference_performed": False,
                    "detector_inference_authorized": False,
                },
                "full_asset_acoustic_gate_passed": False,
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in outputs):
                raise CommonVoiceSileroV55PairingError(
                    "A pairing output appeared while staging."
                )
            staged_candidate.replace(arguments.output_candidate)
            staged_receipt.replace(arguments.output_receipt)
    except (
        CommonVoiceSileroV55PairingError,
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
                "retained_pairs": len(paired_base),
                "candidate_assets": len(candidate),
                "output_candidate": str(arguments.output_candidate),
                "output_receipt": str(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
