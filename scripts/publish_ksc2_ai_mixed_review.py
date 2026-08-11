"""Publish the narrow, explicit single-AI KSC2 mixed transcript review evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from kds.data.ksc2_mixed_review import (
    AI_REVIEW_FIELDS,
    Ksc2MixedReviewError,
    curated_mixed_rows,
    load_candidate_packet,
    write_csv_once,
    write_json_once,
)


def _timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Ksc2MixedReviewError(["reviewed-at must be an ISO-8601 timestamp."]) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish only the explicit high-precision single-AI transcript-review positives from "
            "the hash-pinned KSC2 candidate packet. It performs no language inference."
        )
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-receipt", type=Path, required=True)
    parser.add_argument("--packet-lock", type=Path, required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        if arguments.output_csv == arguments.output_receipt:
            raise Ksc2MixedReviewError(["CSV and receipt output paths must differ."])
        if arguments.output_csv.exists() or arguments.output_receipt.exists():
            raise Ksc2MixedReviewError(
                ["Refusing to overwrite an existing review output or receipt."]
            )
        _timestamp(arguments.reviewed_at)
        packet = load_candidate_packet(
            arguments.packet, arguments.packet_receipt, arguments.packet_lock
        )
        rows = curated_mixed_rows(packet, arguments.reviewed_at)
        csv_hash = write_csv_once(arguments.output_csv, AI_REVIEW_FIELDS, rows)
        component_counts = Counter(row["component"] for row in rows)
        receipt_hash = write_json_once(
            arguments.output_receipt,
            {
                "schema_version": 1,
                "candidate_packet_path": arguments.packet.as_posix(),
                "candidate_packet_sha256": packet.packet_sha256,
                "candidate_packet_receipt_path": arguments.packet_receipt.as_posix(),
                "candidate_packet_receipt_sha256": packet.receipt_sha256,
                "candidate_packet_lock_path": arguments.packet_lock.as_posix(),
                "candidate_packet_lock_sha256": packet.lock_sha256,
                "archive_sha256": rows[0]["archive_sha256"],
                "source_lock_sha256": rows[0]["source_lock_sha256"],
                "review_method": "single_ai_transcript_semantic_review_v1",
                "reviewer": "codex_language_review_v1",
                "reviewed_at": arguments.reviewed_at,
                "candidate_rows": len(packet.rows),
                "confirmed_mixed_rows": len(rows),
                "confirmed_mixed_counts_by_component": dict(sorted(component_counts.items())),
                "output_csv": arguments.output_csv.as_posix(),
                "output_csv_sha256": csv_hash,
                "rule": (
                    "Each published row has explicit Russian and Kazakh transcript-token evidence "
                    "from one AI semantic review. Unlisted candidates remain unknown. This is "
                    "research evidence, not a binary training manifest."
                ),
            },
        )
    except Ksc2MixedReviewError as error:
        print(json.dumps({"status": "error", "issues": list(error.issues)}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_rows": len(packet.rows),
                "confirmed_mixed_rows": len(rows),
                "component_counts": dict(sorted(component_counts.items())),
                "output_csv": str(arguments.output_csv),
                "output_csv_sha256": csv_hash,
                "receipt": str(arguments.output_receipt),
                "receipt_sha256": receipt_hash,
                "explicit_ai_review_decisions": len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
