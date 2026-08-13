"""Publish only exact Stage-D Common Voice/Dialog-RU pairs that passed technical QA."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.dialogs_ru_vits2_candidate import (
    DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID,
    DialogsRuVits2CandidateError,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)


class StageDPairingError(ValueError):
    """Raised when a Stage-D pair would not be completely accounted for."""


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageDPairingError(f"Cannot read {label}: {error}") from error
    if not isinstance(raw, dict):
        raise StageDPairingError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], raw)


def _rows_by_text(rows: list[ManifestRow], label: str) -> dict[str, ManifestRow]:
    by_text = {row.text_id: row for row in rows}
    if len(by_text) != len(rows):
        raise StageDPairingError(f"{label} repeats a text_id.")
    return by_text


def _require_synthesis_report(
    path: Path, *, candidate: Path, raw: Path, text_binding: Path, route_audit: Path
) -> None:
    report = _object(path, "Stage-D synthesis report")
    bindings = {
        "candidate_manifest": candidate,
        "text_binding": text_binding,
        "route_audit": route_audit,
        "output_manifest": raw,
    }
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id") != "stage-d-dialogs-ru-masha-neutral-synthesis-v1"
        or report.get("generated_rows") != 73
        or report.get("exactly_one_synthetic_per_frozen_base") is not True
        or report.get("post_selection_backfill") is not False
        or report.get("reference_audio_or_voice_cloning_used") is not False
        or report.get("detector_inference_performed") is not False
        or report.get("full_asset_acoustic_gate_passed") is not False
    ):
        raise StageDPairingError("Stage-D synthesis report has an invalid protocol state.")
    for name, bound in bindings.items():
        value = report.get(name)
        if (
            not isinstance(value, dict)
            or value.get("path") != bound.as_posix()
            or value.get("sha256") != sha256_file(bound)
        ):
            raise StageDPairingError(f"Stage-D synthesis report has an invalid {name} binding.")


def _rejected_ids(path: Path, *, raw: Path) -> set[str]:
    report = _object(path, "Stage-D technical QA rejection report")
    items = report.get("rejected_rows")
    if (
        report.get("input_manifest") != raw.as_posix()
        or report.get("reused_rows") != 0
        or report.get("published_rows") != 55
        or not isinstance(items, list)
        or len(items) != 18
    ):
        raise StageDPairingError("Stage-D technical QA rejection report has invalid counts.")
    rejected: set[str] = set()
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sample_id"), str)
            or not isinstance(item.get("relative_path"), str)
            or not isinstance(item.get("detail"), str)
            or "insufficient_speech" not in item["detail"]
            or item["sample_id"] in rejected
        ):
            raise StageDPairingError("Stage-D technical QA rejection entry is invalid.")
        rejected.add(item["sample_id"])
    return rejected


def _validate_roles(
    base: list[ManifestRow], raw: list[ManifestRow], ready: list[ManifestRow]
) -> None:
    if len(base) != 73 or any(
        row.split != "test"
        or row.label != "bonafide"
        or row.language != "ru"
        or row.source_name != "common_voice_ru_v24"
        for row in base
    ):
        raise StageDPairingError("Stage-D base must remain 73 frozen Common Voice RU test rows.")
    if len(raw) != 73 or any(
        row.split != "test"
        or row.label != "spoof"
        or row.language != "ru"
        or row.source_name != DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID
        for row in raw
    ):
        raise StageDPairingError("Stage-D raw spoof manifest is not the locked 73-row route.")
    if len(ready) != 55 or any(
        row.split != "test"
        or row.label != "spoof"
        or row.language != "ru"
        or row.source_name != DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID
        or row.codec != "wav"
        for row in ready
    ):
        raise StageDPairingError("Stage-D ready spoof manifest must contain exactly 55 WAV rows.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--raw-spoof-manifest", type=Path, required=True)
    parser.add_argument("--ready-spoof-manifest", type=Path, required=True)
    parser.add_argument("--audio-rejections", type=Path, required=True)
    parser.add_argument("--synthesis-report", type=Path, required=True)
    parser.add_argument("--text-binding", type=Path, required=True)
    parser.add_argument("--route-audit", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-candidate", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    outputs = (arguments.output_candidate, arguments.output_receipt)
    try:
        if (
            len(set(outputs)) != len(outputs)
            or any(path.exists() or not path.parent.is_dir() for path in outputs)
        ):
            raise StageDPairingError(
                "Stage-D candidate outputs must be distinct, new and writable."
            )
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        _require_synthesis_report(
            arguments.synthesis_report,
            candidate=arguments.base_manifest,
            raw=arguments.raw_spoof_manifest,
            text_binding=arguments.text_binding,
            route_audit=arguments.route_audit,
        )
        base = load_manifest(arguments.base_manifest)
        raw = load_manifest(arguments.raw_spoof_manifest)
        ready = load_manifest(arguments.ready_spoof_manifest)
        for rows in (base, raw, ready):
            validate_manifest(rows)
        _validate_roles(base, raw, ready)
        ledger = load_license_ledger(arguments.license_ledger)
        for rows in (base, raw, ready):
            validate_manifest_licenses(rows, ledger)
            require_valid_assets(rows, arguments.data_root)
        rejected = _rejected_ids(arguments.audio_rejections, raw=arguments.raw_spoof_manifest)
        base_by_text = _rows_by_text(base, "Stage-D base")
        raw_by_text = _rows_by_text(raw, "Stage-D raw spoof")
        ready_by_text = _rows_by_text(ready, "Stage-D ready spoof")
        if set(raw_by_text) != set(base_by_text):
            raise StageDPairingError(
                "Stage-D raw spoof rows do not exactly cover frozen base texts."
            )
        if not rejected.issubset({row.sample_id for row in raw}):
            raise StageDPairingError("Technical QA report rejects a non-raw Stage-D sample.")
        expected_ready = {
            row.text_id for row in raw if row.sample_id not in rejected
        }
        if set(ready_by_text) != expected_ready:
            raise StageDPairingError(
                "Ready spoof rows are not raw rows minus exactly the QA rejects."
            )
        paired_base = [base_by_text[text_id] for text_id in sorted(expected_ready)]
        if any(
            base_row.text_hash != ready_by_text[base_row.text_id].text_hash
            or base_row.text_id != ready_by_text[base_row.text_id].text_id
            for base_row in paired_base
        ):
            raise StageDPairingError("A Stage-D ready spoof changes its frozen Common Voice text.")
        pairs = sorted(
            [*paired_base, *(ready_by_text[row.text_id] for row in paired_base)],
            key=lambda row: (row.text_id, row.label),
        )
        validate_manifest(pairs)
        stage = Path(
            tempfile.mkdtemp(prefix=".kds-stage-d-pairs-", dir=arguments.output_receipt.parent)
        )
        try:
            staged_candidate = stage / arguments.output_candidate.name
            write_manifest(staged_candidate, pairs)
            receipt = {
                "schema_version": 1,
                "protocol_id": "stage-d-dialogs-ru-masha-neutral-pairing-v1",
                "created_at": arguments.created_at,
                "inputs": {
                    name: {"path": path.as_posix(), "sha256": sha256_file(path)}
                    for name, path in {
                        "base_manifest": arguments.base_manifest,
                        "raw_spoof_manifest": arguments.raw_spoof_manifest,
                        "ready_spoof_manifest": arguments.ready_spoof_manifest,
                        "audio_rejections": arguments.audio_rejections,
                        "synthesis_report": arguments.synthesis_report,
                        "text_binding": arguments.text_binding,
                        "route_audit": arguments.route_audit,
                        "license_ledger": arguments.license_ledger,
                    }.items()
                },
                "counts": {
                    "frozen_base_rows": len(base),
                    "raw_spoof_rows": len(raw),
                    "technical_qa_rejected_spoof_rows": len(rejected),
                    "ready_spoof_rows": len(ready),
                    "retained_pairs": len(paired_base),
                    "candidate_assets": len(pairs),
                },
                "output_candidate": {
                    "path": arguments.output_candidate.as_posix(),
                    "sha256": sha256_file(staged_candidate),
                    "rows": len(pairs),
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
            staged_receipt = stage / arguments.output_receipt.name
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in outputs):
                raise StageDPairingError("A Stage-D candidate output appeared while staging.")
            staged_candidate.replace(arguments.output_candidate)
            staged_receipt.replace(arguments.output_receipt)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    except (
        DialogsRuVits2CandidateError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        StageDPairingError,
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
                "qa_rejected_pairs": len(rejected),
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
