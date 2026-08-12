"""Publish or evaluate the post-inference two-review FLEURS KK/Silero acoustic gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import AssetValidationError, require_valid_assets, sha256_file
from kds.data.fleurs import FleursIngestionError, verified_fleurs_test_transcripts
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest
from kds.eval.fleurs_kk_acoustic_gate import (
    FleursKkAcousticGateError,
    build_fleurs_kk_acoustic_packet,
    evaluate_fleurs_kk_acoustic_gate,
    read_fleurs_kk_acoustic_reviews,
    write_fleurs_kk_acoustic_packet,
    write_fleurs_kk_acoustic_report,
    write_fleurs_kk_acoustic_review_template,
)


def _publish_packet(arguments: argparse.Namespace) -> dict[str, object]:
    candidate = load_manifest(arguments.candidate_manifest)
    validate_manifest(candidate)
    ledger = load_license_ledger(arguments.license_ledger)
    validate_manifest_licenses(candidate, ledger)
    require_valid_assets(candidate, arguments.audio_root)
    transcripts = verified_fleurs_test_transcripts(
        arguments.release_root, (row for row in candidate if row.label == "bonafide")
    )
    packet = build_fleurs_kk_acoustic_packet(candidate, transcripts)
    write_fleurs_kk_acoustic_packet(arguments.output_packet, packet)
    return {
        "status": "packet_published",
        "assets": len(packet),
        "pairs": len(packet) // 2,
        "candidate_manifest": str(arguments.candidate_manifest),
        "candidate_manifest_sha256": sha256_file(arguments.candidate_manifest),
        "output_packet": str(arguments.output_packet),
        "output_packet_sha256": sha256_file(arguments.output_packet),
        "contains_model_predictions": False,
        "acoustic_quality": "pending_two_independent_reviews_per_asset",
        "metric_status_changed": False,
        "final_or_product_eligible": False,
    }


def _prepare_review(arguments: argparse.Namespace) -> dict[str, object]:
    write_fleurs_kk_acoustic_review_template(
        arguments.output_review, arguments.packet, arguments.reviewer_pseudo_id
    )
    return {
        "status": "review_template_published",
        "assets": 304,
        "packet": str(arguments.packet),
        "packet_sha256": sha256_file(arguments.packet),
        "reviewer_pseudo_id": arguments.reviewer_pseudo_id,
        "output_review": str(arguments.output_review),
        "default_decision": "inconclusive",
    }


def _evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    reviews_by_file = [
        (path, read_fleurs_kk_acoustic_reviews(path)) for path in arguments.reviews
    ]
    reviews = tuple(review for _path, rows in reviews_by_file for review in rows)
    report, results = evaluate_fleurs_kk_acoustic_gate(arguments.packet, reviews)
    report["review_files"] = [
        {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        for path, rows in reviews_by_file
    ]
    write_fleurs_kk_acoustic_report(arguments.output_report, report, results)
    return {
        "status": "gate_evaluated",
        "output_report": str(arguments.output_report),
        "output_report_sha256": sha256_file(arguments.output_report),
        "all_assets_acoustically_verified": report["all_assets_acoustically_verified"],
        "metric_status_changed": False,
        "final_or_product_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    publish = modes.add_parser("publish-packet", help="Create a write-once 304-asset packet.")
    publish.add_argument("--candidate-manifest", type=Path, required=True)
    publish.add_argument("--release-root", type=Path, required=True)
    publish.add_argument("--audio-root", type=Path, required=True)
    publish.add_argument("--license-ledger", type=Path, required=True)
    publish.add_argument("--output-packet", type=Path, required=True)
    template = modes.add_parser(
        "prepare-review", help="Create one reviewer worksheet from the immutable packet."
    )
    template.add_argument("--packet", type=Path, required=True)
    template.add_argument("--reviewer-pseudo-id", required=True)
    template.add_argument("--output-review", type=Path, required=True)
    evaluate = modes.add_parser(
        "evaluate", help="Publish a receipt from exactly two complete independent reviews."
    )
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
        AssetValidationError,
        FleursIngestionError,
        FleursKkAcousticGateError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        ValueError,
    ) as error:
        if isinstance(error, (FleursKkAcousticGateError, LicenseLedgerError, ManifestError)):
            issues = list(error.issues)
        else:
            issues = [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
