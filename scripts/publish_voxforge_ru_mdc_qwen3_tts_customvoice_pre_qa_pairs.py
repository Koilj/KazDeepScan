"""Publish the immutable exact text-matched VoxForge/Qwen pre-QA candidate pairs."""

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
from kds.data.qwen3_tts_customvoice_candidate import QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
from kds.data.voxforge import VOXFORGE_RU_SOURCE_ID

_SYNTHESIS_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-synthesis-v1"
_TECHNICAL_QA_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-technical-qa-v1"
_PAIRING_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-pairing-v1"
_HEX = frozenset("0123456789abcdef")


class VoxForgeQwenPairingError(ValueError):
    """Raised when an immutable VoxForge/Qwen pair would be incomplete or substituted."""


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgeQwenPairingError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise VoxForgeQwenPairingError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise VoxForgeQwenPairingError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_pinned_file(
    values: Mapping[str, object], name: str, expected_path: Path, *, rows: int | None = None
) -> None:
    value = values.get(name)
    if (
        not isinstance(value, Mapping)
        or value.get("path") != expected_path.as_posix()
        or _sha256(value.get("sha256"), f"Receipt {name} SHA-256")
        != sha256_file(expected_path)
        or (rows is not None and value.get("rows") != rows)
    ):
        raise VoxForgeQwenPairingError(f"Receipt has an invalid {name} binding.")


def require_synthesis_receipt(
    path: Path, *, base_manifest: Path, raw_manifest: Path
) -> None:
    """Require the completed fixed-Aiden one-shot layer before it can be paired."""

    receipt = _object(path, "Qwen synthesis receipt")
    base = receipt.get("base_manifest")
    output = receipt.get("output_manifest")
    policy = receipt.get("generation_policy")
    claims = receipt.get("claims")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != _SYNTHESIS_PROTOCOL_ID
        or not isinstance(base, Mapping)
        or base.get("path") != base_manifest.as_posix()
        or base.get("rows") != 79
        or _sha256(base.get("sha256"), "Synthesis base manifest SHA-256")
        != sha256_file(base_manifest)
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
        or not isinstance(claims, Mapping)
        or claims.get("synthetic_audio_generated") is not True
        or claims.get("technical_decode_qa_vad_performed") is not False
        or claims.get("acoustic_review_performed") is not False
        or claims.get("binary_pairing_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise VoxForgeQwenPairingError("Synthesis receipt violates the frozen pair boundary.")


def require_technical_qa_receipt(
    path: Path, *, raw_manifest: Path, ready_manifest: Path, synthesis_receipt: Path
) -> None:
    """Require the fully retained 79-row technical QA layer before pairing."""

    receipt = _object(path, "Qwen technical QA receipt")
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    qa = receipt.get("technical_qa")
    claims = receipt.get("claims")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != _TECHNICAL_QA_PROTOCOL_ID
        or not isinstance(inputs, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(qa, Mapping)
        or not isinstance(claims, Mapping)
    ):
        raise VoxForgeQwenPairingError("Technical QA receipt has an invalid contract.")
    _require_pinned_file(inputs, "raw_manifest", raw_manifest, rows=79)
    _require_pinned_file(inputs, "synthesis_receipt", synthesis_receipt)
    _require_pinned_file(outputs, "ready_manifest", ready_manifest, rows=79)
    rejected = qa.get("rejected_rows")
    if (
        qa.get("raw_rows") != 79
        or qa.get("ready_rows") != 79
        or qa.get("reused_rows") != 0
        or qa.get("resynthesis_or_replacement_or_backfill") is not False
        or rejected != []
        or claims.get("synthetic_audio_generated") is not True
        or claims.get("technical_decode_qa_vad_performed") is not True
        or claims.get("acoustic_review_performed") is not False
        or claims.get("binary_pairing_performed") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
        or claims.get("future_pairing_must_use_only_retained_ready_spoof_rows") is not True
    ):
        raise VoxForgeQwenPairingError("Technical QA receipt violates the frozen pair boundary.")


def _rows_by_text_hash(rows: Sequence[ManifestRow], label: str) -> dict[str, ManifestRow]:
    result = {row.text_hash: row for row in rows}
    if len(result) != len(rows):
        raise VoxForgeQwenPairingError(f"{label} repeats a text hash.")
    return result


def _validate_roles(
    base: Sequence[ManifestRow], raw: Sequence[ManifestRow], ready: Sequence[ManifestRow]
) -> None:
    if (
        len(base) != 79
        or len({row.text_id for row in base}) != 79
        or any(
            row.split != "test"
            or row.label != "bonafide"
            or row.language != "ru"
            or row.source_name != VOXFORGE_RU_SOURCE_ID
            or row.codec != "wav"
            for row in base
        )
    ):
        raise VoxForgeQwenPairingError("Base must remain the exact 79-row VoxForge ready layer.")
    if (
        len(raw) != 79
        or len(ready) != 79
        or any(
            row.split != "test"
            or row.label != "spoof"
            or row.language != "ru"
            or row.source_name != QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID
            or row.voice_id != "qwen3_tts_customvoice:aiden"
            or row.codec != "wav"
            for row in (*raw, *ready)
        )
    ):
        raise VoxForgeQwenPairingError(
            "Qwen raw/ready layers do not retain the locked 79-row route."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--raw-spoof-manifest", type=Path, required=True)
    parser.add_argument("--ready-spoof-manifest", type=Path, required=True)
    parser.add_argument("--synthesis-receipt", type=Path, required=True)
    parser.add_argument("--technical-qa-receipt", type=Path, required=True)
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
            raise VoxForgeQwenPairingError(
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
        require_synthesis_receipt(
            arguments.synthesis_receipt,
            base_manifest=arguments.base_manifest,
            raw_manifest=arguments.raw_spoof_manifest,
        )
        require_technical_qa_receipt(
            arguments.technical_qa_receipt,
            raw_manifest=arguments.raw_spoof_manifest,
            ready_manifest=arguments.ready_spoof_manifest,
            synthesis_receipt=arguments.synthesis_receipt,
        )
        base_by_hash = _rows_by_text_hash(base, "Base manifest")
        raw_by_hash = _rows_by_text_hash(raw, "Raw spoof manifest")
        ready_by_hash = _rows_by_text_hash(ready, "Ready spoof manifest")
        if set(raw_by_hash) != set(base_by_hash) or set(ready_by_hash) != set(base_by_hash):
            raise VoxForgeQwenPairingError(
                "Raw or ready spoof layer does not exactly cover every frozen base text."
            )
        if any(
            base_by_hash[text_hash].text_id != ready_by_hash[text_hash].text_id
            for text_hash in base_by_hash
        ):
            raise VoxForgeQwenPairingError("A retained spoof row changes its paired text ID.")
        paired_base = tuple(base_by_hash[text_hash] for text_hash in sorted(base_by_hash))
        candidate = tuple(
            sorted(
                (*paired_base, *(ready_by_hash[row.text_hash] for row in paired_base)),
                key=lambda row: (row.text_hash, row.label),
            )
        )
        validate_manifest(candidate)
        with tempfile.TemporaryDirectory(
            prefix="kds-voxforge-qwen-pre-qa-pairs-", dir=arguments.output_receipt.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged_candidate = stage / arguments.output_candidate.name
            staged_receipt = stage / arguments.output_receipt.name
            write_manifest(staged_candidate, candidate)
            receipt = {
                "schema_version": 1,
                "protocol_id": _PAIRING_PROTOCOL_ID,
                "created_at": arguments.created_at,
                "inputs": {
                    name: {"path": path.as_posix(), "sha256": sha256_file(path)}
                    for name, path in {
                        "base_manifest": arguments.base_manifest,
                        "raw_spoof_manifest": arguments.raw_spoof_manifest,
                        "ready_spoof_manifest": arguments.ready_spoof_manifest,
                        "synthesis_receipt": arguments.synthesis_receipt,
                        "technical_qa_receipt": arguments.technical_qa_receipt,
                        "license_ledger": arguments.license_ledger,
                    }.items()
                },
                "counts": {
                    "frozen_ready_bona_fide_rows": len(base),
                    "raw_spoof_rows": len(raw),
                    "technical_qa_rejected_spoof_rows": 0,
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
                    "exact_text_hash_and_text_id_match": True,
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
                raise VoxForgeQwenPairingError("A pairing output appeared while staging.")
            staged_candidate.replace(arguments.output_candidate)
            staged_receipt.replace(arguments.output_receipt)
    except (
        LicenseLedgerError,
        ManifestError,
        OSError,
        ValueError,
        VoxForgeQwenPairingError,
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
