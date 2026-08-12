"""Prepare or evaluate the two-listener Stage-C KazakhTTS smoke gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.eval.kazakhtts_acoustic_gate import (
    KazakhTtsAcousticGateError,
    build_kazakhtts_acoustic_packet,
    evaluate_kazakhtts_acoustic_gate,
    read_kazakhtts_reviews,
    write_kazakhtts_acoustic_packet,
    write_kazakhtts_review_template,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare", help="Publish packet and two fail-closed forms.")
    prepare.add_argument("--smoke-report", type=Path, required=True)
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
            packet = build_kazakhtts_acoustic_packet(arguments.smoke_report)
            write_kazakhtts_acoustic_packet(arguments.output_packet, packet)
            write_kazakhtts_review_template(
                arguments.output_reviewer_1, arguments.output_packet, "reviewer_1_REPLACE_ME"
            )
            write_kazakhtts_review_template(
                arguments.output_reviewer_2, arguments.output_packet, "reviewer_2_REPLACE_ME"
            )
            result = {
                "status": "prepared",
                "packet": str(arguments.output_packet),
                "packet_sha256": sha256_file(arguments.output_packet),
                "assets": len(packet),
                "reviewer_forms": [
                    str(arguments.output_reviewer_1),
                    str(arguments.output_reviewer_2),
                ],
            }
        else:
            if arguments.output_report.exists() or not arguments.output_report.parent.is_dir():
                raise KazakhTtsAcousticGateError(
                    ["KazakhTTS gate report must be new with an existing parent."]
                )
            reviews = (
                *read_kazakhtts_reviews(arguments.reviewer_1),
                *read_kazakhtts_reviews(arguments.reviewer_2),
            )
            report = evaluate_kazakhtts_acoustic_gate(arguments.packet, reviews)
            report["evaluated_at"] = arguments.evaluated_at
            report["review_files"] = [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in (arguments.reviewer_1, arguments.reviewer_2)
            ]
            arguments.output_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = {
                "status": "evaluated",
                "output_report": str(arguments.output_report),
                "output_report_sha256": sha256_file(arguments.output_report),
                "approved_input_languages": report["approved_input_languages"],
                "detector_inference_authorized": False,
            }
    except (KazakhTtsAcousticGateError, OSError, ValueError) as error:
        issues = (
            list(error.issues) if isinstance(error, KazakhTtsAcousticGateError) else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
