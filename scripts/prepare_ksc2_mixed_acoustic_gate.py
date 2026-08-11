"""Publish or evaluate the fail-closed acoustic review gate for the frozen 30 KSC2/Silero pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.ksc2_mixed_candidate import Ksc2MixedCandidateError, load_published_mixed_review
from kds.data.manifest import ManifestError, load_manifest, validate_manifest
from kds.eval.mixed_acoustic_gate import (
    MixedAcousticGateError,
    build_mixed_acoustic_gate_packet,
    evaluate_mixed_acoustic_gate,
    load_pair_lock,
    read_mixed_acoustic_gate_reviews,
    write_mixed_acoustic_gate_packet,
    write_mixed_acoustic_gate_report,
    write_mixed_acoustic_gate_review_template,
)


def _publish_packet(arguments: argparse.Namespace) -> dict[str, object]:
    candidate = load_manifest(arguments.candidate_manifest)
    validate_manifest(candidate)
    require_valid_assets(candidate, arguments.audio_root)
    evidence = load_published_mixed_review(arguments.review_csv, arguments.review_receipt)
    pair_lock = load_pair_lock(arguments.pair_lock)
    packet = build_mixed_acoustic_gate_packet(candidate, evidence, pair_lock)
    write_mixed_acoustic_gate_packet(arguments.output_packet, packet)
    return {
        "status": "packet_published",
        "assets": len(packet),
        "pairs": len(packet) // 2,
        "output_packet": str(arguments.output_packet),
        "output_packet_sha256": sha256_file(arguments.output_packet),
        "acoustic_language_preservation": "pending_two_independent_reviews_per_asset",
    }


def _evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    reviews = tuple(
        review
        for path in arguments.reviews
        for review in read_mixed_acoustic_gate_reviews(path)
    )
    report, results = evaluate_mixed_acoustic_gate(arguments.packet, reviews)
    write_mixed_acoustic_gate_report(arguments.output_report, report, results)
    return {
        "status": "gate_evaluated",
        "output_report": str(arguments.output_report),
        "output_report_sha256": sha256_file(arguments.output_report),
        "all_assets_acoustically_verified": report["all_assets_acoustically_verified"],
        "final_or_product_eligible": False,
    }


def _prepare_review(arguments: argparse.Namespace) -> dict[str, object]:
    write_mixed_acoustic_gate_review_template(
        arguments.output_review, arguments.packet, arguments.reviewer_pseudo_id
    )
    return {
        "status": "review_template_published",
        "assets": 60,
        "packet": str(arguments.packet),
        "packet_sha256": sha256_file(arguments.packet),
        "reviewer_pseudo_id": arguments.reviewer_pseudo_id,
        "output_review": str(arguments.output_review),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    publish = modes.add_parser(
        "publish-packet", help="Create a write-once 60-asset listening packet."
    )
    publish.add_argument("--candidate-manifest", type=Path, required=True)
    publish.add_argument("--audio-root", type=Path, required=True)
    publish.add_argument("--review-csv", type=Path, required=True)
    publish.add_argument("--review-receipt", type=Path, required=True)
    publish.add_argument("--pair-lock", type=Path, required=True)
    publish.add_argument("--output-packet", type=Path, required=True)
    template = modes.add_parser(
        "prepare-review", help="Create one independent reviewer worksheet from a packet."
    )
    template.add_argument("--packet", type=Path, required=True)
    template.add_argument("--reviewer-pseudo-id", required=True)
    template.add_argument("--output-review", type=Path, required=True)
    evaluate = modes.add_parser(
        "evaluate", help="Evaluate complete independent reviews against a packet."
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
        MixedAcousticGateError,
        Ksc2MixedCandidateError,
        ManifestError,
        OSError,
        ValueError,
    ) as error:
        if isinstance(error, (MixedAcousticGateError, Ksc2MixedCandidateError, ManifestError)):
            issues = list(error.issues)
        else:
            issues = [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
