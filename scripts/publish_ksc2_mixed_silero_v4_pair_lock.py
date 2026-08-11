"""Pin every narrow KSC2/Silero research pair back to its explicit transcript evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.ksc2_mixed_candidate import (
    Ksc2MixedCandidateError,
    load_published_mixed_review,
)
from kds.data.ksc2_mixed_silero_v4 import build_paired_mixed_candidate_rows
from kds.data.manifest import ManifestError, load_manifest, validate_manifest


def _rejected_ids(path: Path, label: str) -> set[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        rejected = value["rejected_rows"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise Ksc2MixedCandidateError([f"Cannot read {label}: {error}"]) from error
    if not isinstance(rejected, list):
        raise Ksc2MixedCandidateError([f"{label} rejected_rows must be an array."])
    identifiers: set[str] = set()
    for item in rejected:
        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
            raise Ksc2MixedCandidateError([f"{label} has an invalid sample_id."])
        identifier = item["sample_id"]
        if identifier in identifiers:
            raise Ksc2MixedCandidateError([f"{label} has a duplicate sample_id."])
        identifiers.add(identifier)
    return identifiers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a write-once provenance lock for the narrow input-pinned KSC2/Silero pairs. "
            "It does not certify acoustic language preservation."
        )
    )
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--base-ready-manifest", type=Path, required=True)
    parser.add_argument("--raw-spoof-manifest", type=Path, required=True)
    parser.add_argument("--ready-spoof-manifest", type=Path, required=True)
    parser.add_argument("--text-rejection-report", type=Path, required=True)
    parser.add_argument("--audio-rejection-report", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise Ksc2MixedCandidateError(
                ["Pair-lock output must be new and have an existing parent."]
            )
        evidence = load_published_mixed_review(arguments.review_csv, arguments.review_receipt)
        evidence_by_id = {item.annotation_id: item for item in evidence}
        base = load_manifest(arguments.base_ready_manifest)
        raw = load_manifest(arguments.raw_spoof_manifest)
        ready = load_manifest(arguments.ready_spoof_manifest)
        candidate = load_manifest(arguments.candidate_manifest)
        for rows in (base, raw, ready, candidate):
            validate_manifest(rows)
        expected = build_paired_mixed_candidate_rows(
            base_rows=base,
            raw_spoof_rows=raw,
            ready_spoof_rows=ready,
            text_rejected_base_ids=_rejected_ids(
                arguments.text_rejection_report, "text rejections"
            ),
            audio_rejected_spoof_ids=_rejected_ids(
                arguments.audio_rejection_report, "audio rejections"
            ),
        )
        if candidate != expected:
            raise Ksc2MixedCandidateError(
                ["Candidate manifest differs from its exact accounted pairs."]
            )
        pair_count = len(expected) // 2
        base_by_text = {row.text_id: row for row in expected[:pair_count]}
        spoof_by_text = {row.text_id: row for row in expected[pair_count:]}
        pairs: list[dict[str, str]] = []
        for text_id, base_row in sorted(base_by_text.items()):
            evidence_row = evidence_by_id.get(base_row.sample_id)
            spoof_row = spoof_by_text.get(text_id)
            if evidence_row is None or spoof_row is None:
                raise Ksc2MixedCandidateError(["Pair cannot be linked to published KSC2 evidence."])
            pairs.append(
                {
                    "annotation_id": evidence_row.annotation_id,
                    "component": evidence_row.component,
                    "text_hash": base_row.text_hash,
                    "bonafide_audio_sha256": base_row.sha256,
                    "spoof_audio_sha256": spoof_row.sha256,
                    "ru_evidence_token_indices": evidence_row.ru_evidence_token_indices,
                    "ru_evidence_tokens": evidence_row.ru_evidence_tokens,
                    "kk_evidence_token_indices": evidence_row.kk_evidence_token_indices,
                    "kk_evidence_tokens": evidence_row.kk_evidence_tokens,
                }
            )
        payload = json.dumps(
            {
                "schema_version": 1,
                "review_csv": arguments.review_csv.as_posix(),
                "review_csv_sha256": sha256_file(arguments.review_csv),
                "review_receipt": arguments.review_receipt.as_posix(),
                "review_receipt_sha256": sha256_file(arguments.review_receipt),
                "base_ready_manifest": arguments.base_ready_manifest.as_posix(),
                "base_ready_manifest_sha256": sha256_file(arguments.base_ready_manifest),
                "raw_spoof_manifest": arguments.raw_spoof_manifest.as_posix(),
                "raw_spoof_manifest_sha256": sha256_file(arguments.raw_spoof_manifest),
                "ready_spoof_manifest": arguments.ready_spoof_manifest.as_posix(),
                "ready_spoof_manifest_sha256": sha256_file(arguments.ready_spoof_manifest),
                "candidate_manifest": arguments.candidate_manifest.as_posix(),
                "candidate_manifest_sha256": sha256_file(arguments.candidate_manifest),
                "pair_count": pair_count,
                "pairs": pairs,
                "rule": (
                    "Every pair is pinned to explicit transcript evidence. "
                    "Synthetic language provenance is intended input text only, "
                    "not acoustic preservation certification."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        with arguments.output.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    except (Ksc2MixedCandidateError, ManifestError, OSError, ValueError) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (Ksc2MixedCandidateError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {"status": "ok", "pair_count": pair_count, "output": str(arguments.output)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
