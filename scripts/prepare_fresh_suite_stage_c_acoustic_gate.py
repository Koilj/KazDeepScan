"""Prepare or evaluate the full two-review Stage-C KazakhTTS acoustic gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest
from kds.eval.kazakhtts_full_acoustic_gate import (
    KazakhTtsFullAcousticGateError,
    build_kazakhtts_full_acoustic_packet,
    evaluate_kazakhtts_full_acoustic_gate,
    read_kazakhtts_full_reviews,
    write_kazakhtts_full_acoustic_packet,
    write_kazakhtts_full_acoustic_report,
    write_kazakhtts_full_review_template,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare", help="Publish packet and two fail-closed forms.")
    prepare.add_argument("--candidate-manifest", type=Path, required=True)
    prepare.add_argument("--pairing-receipt", type=Path, required=True)
    prepare.add_argument("--normalization-plan", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--license-ledger", type=Path, required=True)
    prepare.add_argument("--output-packet", type=Path, required=True)
    prepare.add_argument("--output-reviewer-1", type=Path, required=True)
    prepare.add_argument("--output-reviewer-2", type=Path, required=True)
    evaluate = modes.add_parser("evaluate", help="Evaluate two completed review forms.")
    evaluate.add_argument("--packet", type=Path, required=True)
    evaluate.add_argument("--reviewer-1", type=Path, required=True)
    evaluate.add_argument("--reviewer-2", type=Path, required=True)
    evaluate.add_argument("--evaluated-at", required=True)
    evaluate.add_argument("--output-report", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.mode == "prepare":
            candidate = load_manifest(arguments.candidate_manifest)
            validate_manifest(candidate)
            validate_manifest_licenses(
                candidate, load_license_ledger(arguments.license_ledger)
            )
            require_valid_assets(candidate, arguments.data_root)
            packet = build_kazakhtts_full_acoustic_packet(
                candidate_rows=candidate,
                candidate_path=arguments.candidate_manifest,
                pairing_receipt_path=arguments.pairing_receipt,
                normalization_plan_path=arguments.normalization_plan,
                data_root=arguments.data_root,
            )
            write_kazakhtts_full_acoustic_packet(arguments.output_packet, packet)
            write_kazakhtts_full_review_template(
                arguments.output_reviewer_1,
                arguments.output_packet,
                "reviewer_1_REPLACE_ME",
            )
            write_kazakhtts_full_review_template(
                arguments.output_reviewer_2,
                arguments.output_packet,
                "reviewer_2_REPLACE_ME",
            )
            result = {
                "status": "prepared",
                "assets": len(packet),
                "packet": str(arguments.output_packet),
                "packet_sha256": sha256_file(arguments.output_packet),
                "reviewer_forms": [
                    str(arguments.output_reviewer_1),
                    str(arguments.output_reviewer_2),
                ],
                "detector_inference_authorized": False,
            }
        else:
            reviews = (
                *read_kazakhtts_full_reviews(arguments.reviewer_1),
                *read_kazakhtts_full_reviews(arguments.reviewer_2),
            )
            report = evaluate_kazakhtts_full_acoustic_gate(arguments.packet, reviews)
            report["evaluated_at"] = arguments.evaluated_at
            report["review_files"] = [
                {"path": path.as_posix(), "sha256": sha256_file(path)}
                for path in (arguments.reviewer_1, arguments.reviewer_2)
            ]
            write_kazakhtts_full_acoustic_report(arguments.output_report, report)
            result = {
                "status": "evaluated",
                "output_report": str(arguments.output_report),
                "output_report_sha256": sha256_file(arguments.output_report),
                "all_assets_acoustically_verified": report[
                    "all_assets_acoustically_verified"
                ],
                "detector_inference_authorized": False,
            }
    except (
        KazakhTtsFullAcousticGateError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        ValueError,
    ) as error:
        issues = (
            list(error.issues)
            if isinstance(
                error,
                (KazakhTtsFullAcousticGateError, LicenseLedgerError, ManifestError),
            )
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
