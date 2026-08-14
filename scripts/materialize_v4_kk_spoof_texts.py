"""Extract and hash-bind the frozen KSC2 text inputs for v4 KK spoof synthesis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.ksc2 import Ksc2AuditError, extract_ksc2_selected_transcripts
from kds.data.v4_synthesis import (
    V4SynthesisError,
    load_v4_kk_spoof_candidates,
    load_verified_v4_transcript,
)

V4_TEXT_INVENTORY_FIELDS = (
    "selection_rank",
    "target_state",
    "candidate_id",
    "pair_id",
    "generator_route_id",
    "generator_family",
    "source_component",
    "archive_transcript_member",
    "transcript_relative_path",
    "transcript_file_sha256",
    "transcript_size_bytes",
    "text_hash",
    "canonical_text_hash",
    "normalized_utf8_bytes",
    "normalized_characters",
    "status",
)


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4SynthesisError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4SynthesisError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4SynthesisError(f"{label} is not a lowercase SHA-256 digest.")
    return value


def _progress(completed: int, total: int, item: str) -> None:
    print(
        json.dumps(
            {
                "status": "progress",
                "stage": "ksc2_transcript_extraction",
                "completed": completed,
                "total": total,
                "item": item,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--governance-receipt", type=Path, required=True)
    parser.add_argument("--source-decode-receipt", type=Path, required=True)
    parser.add_argument("--ksc2-audit", type=Path, required=True)
    parser.add_argument("--ksc2-parts-directory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-inventory", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    published_raw = False
    published_outputs: list[tuple[Path, Path]] = []
    try:
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        data_root = arguments.data_root.resolve(strict=True)
        destination = arguments.output_directory.resolve()
        destination.relative_to(data_root / "raw" / "v4")
        outputs = (arguments.output_inventory, arguments.output_receipt)
        if (
            destination.exists()
            or not destination.parent.is_dir()
            or len(set(outputs)) != len(outputs)
            or any(path.exists() or not path.parent.is_dir() for path in outputs)
        ):
            raise V4SynthesisError("Unsafe v4 KK transcript output destinations.")
        candidates = load_v4_kk_spoof_candidates(
            arguments.candidate_csv,
            arguments.governance_receipt,
            arguments.source_decode_receipt,
        )
        audit = _json_object(arguments.ksc2_audit, "KSC2 audit")
        compressed_hash = _sha256(audit.get("compressed_sha256"), "KSC2 compressed archive")
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-kk-texts-", dir=destination.parent
        ) as stage_name:
            stage = Path(stage_name)
            payload = stage / "payload"
            extracted = extract_ksc2_selected_transcripts(
                arguments.ksc2_parts_directory,
                payload,
                selected_members=frozenset(row.transcript_member for row in candidates),
                expected_compressed_sha256=compressed_hash,
                progress_callback=_progress,
            )
            extracted_by_member = {item.archive_member: item for item in extracted}
            staged_inventory = stage / arguments.output_inventory.name
            with staged_inventory.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=V4_TEXT_INVENTORY_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for candidate in candidates:
                    item = extracted_by_member[candidate.transcript_member]
                    text = load_verified_v4_transcript(candidate, payload)
                    relative_path = (
                        destination.relative_to(data_root) / Path(item.relative_path)
                    ).as_posix()
                    writer.writerow(
                        {
                            "selection_rank": candidate.selection_rank,
                            "target_state": candidate.target_state,
                            "candidate_id": candidate.candidate_id,
                            "pair_id": candidate.pair_id,
                            "generator_route_id": candidate.generator_route_id,
                            "generator_family": candidate.generator_family,
                            "source_component": candidate.source_component,
                            "archive_transcript_member": candidate.transcript_member,
                            "transcript_relative_path": relative_path,
                            "transcript_file_sha256": item.sha256,
                            "transcript_size_bytes": item.size_bytes,
                            "text_hash": candidate.text_hash,
                            "canonical_text_hash": candidate.canonical_text_hash,
                            "normalized_utf8_bytes": len(text.encode("utf-8")),
                            "normalized_characters": len(text),
                            "status": "verified_for_synthesis",
                        }
                    )
            receipt = {
                "schema_version": 1,
                "protocol_id": "xlsr-sls-model-v4-kk-spoof-text-materialization-v1",
                "created_at": arguments.created_at,
                "state": "kk_spoof_texts_verified_synthesis_pending",
                "bindings": {
                    "candidate_csv": {
                        "path": arguments.candidate_csv.as_posix(),
                        "sha256": sha256_file(arguments.candidate_csv),
                    },
                    "selection_governance": {
                        "path": arguments.governance_receipt.as_posix(),
                        "sha256": sha256_file(arguments.governance_receipt),
                    },
                    "source_decode_receipt": {
                        "path": arguments.source_decode_receipt.as_posix(),
                        "sha256": sha256_file(arguments.source_decode_receipt),
                    },
                    "ksc2_audit": {
                        "path": arguments.ksc2_audit.as_posix(),
                        "sha256": sha256_file(arguments.ksc2_audit),
                        "compressed_sha256": compressed_hash,
                    },
                },
                "outputs": {
                    "transcript_directory": destination.relative_to(data_root.parent).as_posix(),
                    "inventory": {
                        "path": arguments.output_inventory.as_posix(),
                        "sha256": sha256_file(staged_inventory),
                        "rows": len(candidates),
                    },
                },
                "accounting": {
                    "verified_transcripts": len(candidates),
                    "unique_transcript_members": len(
                        {candidate.transcript_member for candidate in candidates}
                    ),
                    "unique_text_hashes": len(
                        {candidate.canonical_text_hash for candidate in candidates}
                    ),
                    "route_rows": {
                        route: sum(
                            candidate.generator_route_id == route for candidate in candidates
                        )
                        for route in sorted(
                            {candidate.generator_route_id for candidate in candidates}
                        )
                    },
                },
                "claims": {
                    "full_ksc2_archive_hash_and_crc_reverified": True,
                    "exact_transcript_allow_list_extracted": True,
                    "normalized_text_hashes_verified": True,
                    "synthesis_performed": False,
                    "training_authorized": False,
                    "new_dataset_search_performed": False,
                },
                "next_gate": (
                    "execute the separately frozen four-route v4 KK synthesis plan; retain "
                    "complete success/failure accounting and do not train before audio QA"
                ),
            }
            staged_receipt = stage / arguments.output_receipt.name
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            payload.replace(destination)
            published_raw = True
            for staged, output in (
                (staged_inventory, arguments.output_inventory),
                (staged_receipt, arguments.output_receipt),
            ):
                os.link(staged, output)
                published_outputs.append((output, staged))
        # Revalidate normalized content after publication, independent of staging paths.
        for candidate in candidates:
            text = load_verified_v4_transcript(candidate, destination)
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != candidate.text_hash:
                raise V4SynthesisError("Published v4 transcript verification failed.")
    except (Ksc2AuditError, OSError, ValueError, V4SynthesisError) as error:
        for output, staged in reversed(published_outputs):
            try:
                if output.samefile(staged):
                    output.unlink()
            except OSError:
                pass
        if published_raw:
            shutil.rmtree(arguments.output_directory, ignore_errors=True)
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(candidates),
                "inventory": arguments.output_inventory.as_posix(),
                "inventory_sha256": sha256_file(arguments.output_inventory),
                "receipt": arguments.output_receipt.as_posix(),
                "receipt_sha256": sha256_file(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
