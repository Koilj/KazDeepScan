"""Publish or evaluate the two-review acoustic/language gate for VoxForge/Qwen pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.voxforge import load_voxforge_ru_metadata
from kds.eval.voxforge_metadata_screen import voxforge_metadata_identity
from kds.eval.voxforge_qwen_acoustic_gate import (
    AcousticReview,
    VoxForgeQwenAcousticGateError,
    build_packet,
    evaluate,
    read_reviews,
    write_packet,
    write_report,
    write_review_template,
)

_PAIRING_PROTOCOL_ID = "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-pre-qa-pairing-v1"


def _receipt(path: Path) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxForgeQwenAcousticGateError(f"Cannot read pairing receipt: {error}") from error
    if not isinstance(value, dict):
        raise VoxForgeQwenAcousticGateError("Pairing receipt must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_pairing_receipt(path: Path, candidate: Path) -> None:
    receipt = _receipt(path)
    output = receipt.get("output_candidate")
    counts = receipt.get("counts")
    rule = receipt.get("decision_rule")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != _PAIRING_PROTOCOL_ID
        or not isinstance(output, Mapping)
        or output.get("path") != candidate.as_posix()
        or output.get("sha256") != sha256_file(candidate)
        or output.get("rows") != 158
        or not isinstance(counts, Mapping)
        or counts.get("retained_pairs") != 79
        or counts.get("technical_qa_rejected_spoof_rows") != 0
        or not isinstance(rule, Mapping)
        or rule.get("exact_text_hash_and_text_id_match") is not True
        or rule.get("qa_reject_excludes_entire_pair") is not True
        or rule.get("post_selection_backfill") is not False
        or rule.get("resynthesis_after_qa") is not False
        or rule.get("detector_inference_performed") is not False
        or rule.get("detector_inference_authorized") is not False
        or receipt.get("full_asset_acoustic_gate_passed") is not False
    ):
        raise VoxForgeQwenAcousticGateError("Pairing receipt is invalid or changed.")


def _literal_transcripts(archive: Path, candidate: list[ManifestRow]) -> dict[str, str]:
    records = load_voxforge_ru_metadata(archive)
    by_id = {voxforge_metadata_identity(record).sample_id: record for record in records}
    transcripts: dict[str, str] = {}
    for row in candidate:
        if row.label != "bonafide":
            continue
        source = by_id.get(row.sample_id)
        if source is None or (
            hashlib.sha256(source.prompt_text.encode("utf-8")).hexdigest() != row.text_hash
        ):
            raise VoxForgeQwenAcousticGateError(
                f"Pinned archive literal text differs for base row {row.sample_id!r}."
            )
        transcripts[row.sample_id] = source.prompt_text
    return transcripts


def _publish_packet(arguments: argparse.Namespace) -> dict[str, object]:
    candidate = load_manifest(arguments.candidate_manifest)
    validate_manifest(candidate)
    ledger = load_license_ledger(arguments.license_ledger)
    validate_manifest_licenses(candidate, ledger)
    require_valid_assets(candidate, arguments.audio_root)
    _require_pairing_receipt(arguments.pairing_receipt, arguments.candidate_manifest)
    packet = build_packet(candidate, _literal_transcripts(arguments.voxforge_archive, candidate))
    write_packet(arguments.output_packet, packet)
    return {
        "status": "packet_published",
        "assets": len(packet),
        "pairs": len(packet) // 2,
        "output_packet": str(arguments.output_packet),
        "output_packet_sha256": sha256_file(arguments.output_packet),
        "acoustic_gate": "pending_two_independent_reviews_per_asset",
        "detector_inference_performed": False,
    }


def _prepare_review(arguments: argparse.Namespace) -> dict[str, object]:
    write_review_template(arguments.output_review, arguments.packet, arguments.reviewer_pseudo_id)
    return {
        "status": "review_template_published",
        "assets": 158,
        "packet": str(arguments.packet),
        "packet_sha256": sha256_file(arguments.packet),
        "reviewer_pseudo_id": arguments.reviewer_pseudo_id,
        "output_review": str(arguments.output_review),
    }


def _evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    review_files: list[dict[str, object]] = []
    all_reviews: list[AcousticReview] = []
    for path in arguments.reviews:
        reviews = read_reviews(path)
        reviewer_ids = sorted({review.reviewer_pseudo_id for review in reviews})
        if len(reviews) != 158 or len(reviewer_ids) != 1:
            raise VoxForgeQwenAcousticGateError(
                "Each review must contain 158 decisions from exactly one reviewer."
            )
        review_files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(reviews),
                "reviewer_pseudo_id": reviewer_ids[0],
            }
        )
        all_reviews.extend(reviews)
    if len(review_files) != 2 or len({item["reviewer_pseudo_id"] for item in review_files}) != 2:
        raise VoxForgeQwenAcousticGateError(
            "Gate requires exactly two forms with distinct reviewer pseudo-IDs."
        )
    report, results = evaluate(arguments.packet, all_reviews)
    report["review_files"] = review_files
    write_report(arguments.output_report, report, results)
    return {
        "status": "gate_evaluated",
        "output_report": str(arguments.output_report),
        "output_report_sha256": sha256_file(arguments.output_report),
        "all_assets_acoustically_verified": report["all_assets_acoustically_verified"],
        "detector_inference_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    publish = modes.add_parser("publish-packet", help="Create a write-once 158-asset packet.")
    publish.add_argument("--candidate-manifest", type=Path, required=True)
    publish.add_argument("--pairing-receipt", type=Path, required=True)
    publish.add_argument("--voxforge-archive", type=Path, required=True)
    publish.add_argument("--audio-root", type=Path, required=True)
    publish.add_argument("--license-ledger", type=Path, required=True)
    publish.add_argument("--output-packet", type=Path, required=True)
    template = modes.add_parser("prepare-review", help="Create one independent review worksheet.")
    template.add_argument("--packet", type=Path, required=True)
    template.add_argument("--reviewer-pseudo-id", required=True)
    template.add_argument("--output-review", type=Path, required=True)
    evaluate_mode = modes.add_parser("evaluate", help="Evaluate two completed review worksheets.")
    evaluate_mode.add_argument("--packet", type=Path, required=True)
    evaluate_mode.add_argument("--reviews", type=Path, required=True, action="append")
    evaluate_mode.add_argument("--output-report", type=Path, required=True)
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
        OSError,
        ValueError,
        VoxForgeQwenAcousticGateError,
    ) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (LicenseLedgerError, ManifestError, VoxForgeQwenAcousticGateError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
