#!/usr/bin/env python3
"""Publish or evaluate a two-review acoustic language gate for locked ToneSpeak WAVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.tone_speak import (
    TONE_SPEAK_SOURCE_ID,
    ToneSpeakAuditError,
    audit_tone_speak_release,
    load_tone_speak_records,
)
from kds.eval.tone_speak_acoustic_gate import (
    ToneSpeakAcousticGateError,
    ToneSpeakAcousticReview,
    build_tone_speak_acoustic_packet,
    evaluate_tone_speak_acoustic_gate,
    read_tone_speak_acoustic_reviews,
    write_tone_speak_acoustic_packet,
    write_tone_speak_acoustic_report,
    write_tone_speak_acoustic_review_template,
)


def _validate_ready_receipt(path: Path, candidate_manifest: Path) -> None:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToneSpeakAcousticGateError(f"Cannot read ToneSpeak ready receipt: {path}") from error
    if (
        receipt.get("source_id") != TONE_SPEAK_SOURCE_ID
        or receipt.get("ready_manifest") != str(candidate_manifest)
        or receipt.get("ready_manifest_sha256") != sha256_file(candidate_manifest)
        or receipt.get("ready_rows") != 100
        or receipt.get("rejected_rows") != 0
    ):
        raise ToneSpeakAcousticGateError(
            "ToneSpeak ready receipt does not bind an intact 100-asset ready candidate."
        )


def _source_transcripts(
    artifact_root: Path, candidate_rows: list[ManifestRow]
) -> dict[str, str]:
    audit_tone_speak_release(artifact_root)
    source_by_sample = {
        f"{TONE_SPEAK_SOURCE_ID}:{Path(record.embedded_path).stem}": record
        for record in load_tone_speak_records(artifact_root, source_split="validation")
    }
    transcripts: dict[str, str] = {}
    for row in candidate_rows:
        source = source_by_sample.get(row.sample_id)
        if (
            source is None
            or row.text_hash != source.text_hash
            or row.voice_id != f"{TONE_SPEAK_SOURCE_ID}:voice:{source.voice_name}"
        ):
            raise ToneSpeakAcousticGateError(
                "ToneSpeak ready row does not bind source validation provenance: "
                f"{row.sample_id!r}."
            )
        transcripts[row.sample_id] = source.text
    return transcripts


def _publish_packet(arguments: argparse.Namespace) -> dict[str, object]:
    candidate = load_manifest(arguments.candidate_manifest)
    validate_manifest(candidate)
    ledger = load_license_ledger(arguments.license_ledger)
    validate_manifest_licenses(candidate, ledger)
    require_valid_assets(candidate, arguments.audio_root)
    _validate_ready_receipt(arguments.ready_receipt, arguments.candidate_manifest)
    packet = build_tone_speak_acoustic_packet(
        candidate, _source_transcripts(arguments.artifact_root, candidate)
    )
    write_tone_speak_acoustic_packet(arguments.output_packet, packet)
    return {
        "status": "packet_published",
        "assets": len(packet),
        "output_packet": str(arguments.output_packet),
        "output_packet_sha256": sha256_file(arguments.output_packet),
        "acoustic_language_preservation": "pending_two_independent_reviews_per_asset",
        "final_or_product_eligible": False,
    }


def _prepare_review(arguments: argparse.Namespace) -> dict[str, object]:
    write_tone_speak_acoustic_review_template(
        arguments.output_review, arguments.packet, arguments.reviewer_pseudo_id
    )
    return {
        "status": "review_template_published",
        "assets": 100,
        "packet": str(arguments.packet),
        "packet_sha256": sha256_file(arguments.packet),
        "reviewer_pseudo_id": arguments.reviewer_pseudo_id,
        "output_review": str(arguments.output_review),
    }


def _evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    review_files: list[dict[str, object]] = []
    reviews: list[ToneSpeakAcousticReview] = []
    for path in arguments.reviews:
        file_reviews = read_tone_speak_acoustic_reviews(path)
        reviewer_ids = sorted({review.reviewer_pseudo_id for review in file_reviews})
        if len(file_reviews) != 100 or len(reviewer_ids) != 1:
            raise ToneSpeakAcousticGateError(
                "Each ToneSpeak review file must contain exactly 100 decisions from one reviewer."
            )
        review_files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(file_reviews),
                "reviewer_pseudo_id": reviewer_ids[0],
            }
        )
        reviews.extend(file_reviews)
    if len(review_files) != 2 or len({item["reviewer_pseudo_id"] for item in review_files}) != 2:
        raise ToneSpeakAcousticGateError(
            "ToneSpeak gate requires exactly two review files with distinct reviewer pseudo-IDs."
        )
    report, results = evaluate_tone_speak_acoustic_gate(arguments.packet, reviews)
    report["review_files"] = review_files
    write_tone_speak_acoustic_report(arguments.output_report, report, results)
    return {
        "status": "gate_evaluated",
        "output_report": str(arguments.output_report),
        "output_report_sha256": sha256_file(arguments.output_report),
        "all_assets_acoustically_verified": report["all_assets_acoustically_verified"],
        "final_or_product_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    publish = modes.add_parser("publish-packet", help="Create a write-once 100-asset packet.")
    publish.add_argument("--candidate-manifest", type=Path, required=True)
    publish.add_argument("--ready-receipt", type=Path, required=True)
    publish.add_argument("--artifact-root", type=Path, required=True)
    publish.add_argument("--audio-root", type=Path, required=True)
    publish.add_argument("--license-ledger", type=Path, required=True)
    publish.add_argument("--output-packet", type=Path, required=True)
    template = modes.add_parser(
        "prepare-review", help="Create one independent reviewer worksheet from a packet."
    )
    template.add_argument("--packet", type=Path, required=True)
    template.add_argument("--reviewer-pseudo-id", required=True)
    template.add_argument("--output-review", type=Path, required=True)
    evaluate = modes.add_parser("evaluate", help="Evaluate complete reviews against a packet.")
    evaluate.add_argument("--packet", type=Path, required=True)
    evaluate.add_argument("--reviews", type=Path, required=True, action="append")
    evaluate.add_argument("--output-report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.mode == "publish-packet":
            result = _publish_packet(arguments)
        elif arguments.mode == "prepare-review":
            result = _prepare_review(arguments)
        else:
            result = _evaluate(arguments)
    except (
        LicenseLedgerError,
        ManifestError,
        ToneSpeakAcousticGateError,
        ToneSpeakAuditError,
        OSError,
        ValueError,
    ) as error:
        if isinstance(error, (LicenseLedgerError, ManifestError, ToneSpeakAcousticGateError)):
            issues = list(error.issues)
        else:
            issues = [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
