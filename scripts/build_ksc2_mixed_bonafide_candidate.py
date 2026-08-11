"""QA/VAD-prepare the narrow published KSC2 mixed evidence as bona-fide candidates."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import require_valid_assets, resolve_asset_path, sha256_file
from kds.data.ksc2_mixed_candidate import (
    Ksc2MixedAudioInfo,
    Ksc2MixedCandidateError,
    Ksc2MixedEvidenceRow,
    load_published_mixed_review,
    mixed_bonafide_rows,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, write_manifest
from kds.data.preprocess import preprocess_rows


def _audio_info(
    data_root: Path, evidence: Sequence[Ksc2MixedEvidenceRow]
) -> dict[str, Ksc2MixedAudioInfo]:
    result: dict[str, Ksc2MixedAudioInfo] = {}
    for item in evidence:
        path = resolve_asset_path(data_root, item.audio_relative_path)
        if not path.is_file() or sha256_file(path) != item.audio_sha256:
            raise Ksc2MixedCandidateError(
                [f"KSC2 reviewed audio is missing or changed: {item.annotation_id}."]
            )
        try:
            info = sf.info(str(path))
        except RuntimeError as error:
            raise Ksc2MixedCandidateError(
                [f"Cannot inspect KSC2 reviewed audio {item.annotation_id}: {error}"]
            ) from error
        if info.duration <= 0 or info.samplerate <= 0 or not info.format:
            raise Ksc2MixedCandidateError(
                [f"KSC2 reviewed audio has invalid metadata: {item.annotation_id}."]
            )
        result[item.annotation_id] = Ksc2MixedAudioInfo(
            duration_s=float(info.duration),
            original_sr=int(info.samplerate),
            codec=str(info.format).lower(),
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish raw and QA/VAD-ready KSC2 mixed bona-fide candidates from the fixed "
            "32-row AI review. It creates neither spoof assets nor a binary final layer."
        )
    )
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--rejection-report", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args(argv)

    try:
        output_paths = (
            arguments.raw_manifest,
            arguments.ready_manifest,
            arguments.rejection_report,
        )
        if len(set(output_paths)) != len(output_paths):
            raise Ksc2MixedCandidateError(["Output paths must be distinct."])
        if any(path.exists() for path in output_paths):
            raise Ksc2MixedCandidateError(["Refusing to overwrite a KSC2 mixed candidate output."])
        if any(not path.parent.is_dir() for path in output_paths):
            raise Ksc2MixedCandidateError(["Every output parent directory must already exist."])
        data_root = arguments.data_root.resolve(strict=True)
        evidence = load_published_mixed_review(arguments.review_csv, arguments.review_receipt)
        raw_rows = mixed_bonafide_rows(
            evidence, _audio_info(data_root, evidence), created_at=arguments.created_at
        )
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(raw_rows, ledger)
        require_valid_assets(raw_rows, data_root)
        prepared = preprocess_rows(
            raw_rows,
            data_root,
            AudioPreparationPipeline(),
            allow_rejections=True,
        )
        validate_manifest_licenses(prepared.processed_rows, ledger)
        require_valid_assets(prepared.processed_rows, data_root)
        with tempfile.TemporaryDirectory(
            prefix="kds-ksc2-mixed-manifest-", dir=arguments.raw_manifest.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged_raw = stage / arguments.raw_manifest.name
            staged_ready = stage / arguments.ready_manifest.name
            staged_rejections = stage / arguments.rejection_report.name
            write_manifest(staged_raw, raw_rows)
            write_manifest(staged_ready, prepared.processed_rows)
            staged_rejections.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "review_csv": arguments.review_csv.as_posix(),
                        "review_csv_sha256": sha256_file(arguments.review_csv),
                        "review_receipt": arguments.review_receipt.as_posix(),
                        "review_receipt_sha256": sha256_file(arguments.review_receipt),
                        "raw_manifest": arguments.raw_manifest.as_posix(),
                        "raw_rows": len(raw_rows),
                        "ready_rows": len(prepared.processed_rows),
                        "rejected_rows": [
                            {
                                "sample_id": issue.sample_id,
                                "relative_path": issue.relative_path,
                                "reason": issue.detail,
                            }
                            for issue in prepared.issues
                        ],
                        "rule": (
                            "Narrow single-AI transcript-review bona-fide candidates only; "
                            "not a binary training or final-test manifest."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in output_paths):
                raise Ksc2MixedCandidateError(["An output appeared while publication was staging."])
            staged_raw.replace(arguments.raw_manifest)
            staged_ready.replace(arguments.ready_manifest)
            staged_rejections.replace(arguments.rejection_report)
    except (
        Ksc2MixedCandidateError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        ValueError,
    ) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (Ksc2MixedCandidateError, LicenseLedgerError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "raw_rows": len(raw_rows),
                "ready_rows": len(prepared.processed_rows),
                "rejected_rows": len(prepared.issues),
                "raw_manifest": str(arguments.raw_manifest),
                "ready_manifest": str(arguments.ready_manifest),
                "rejection_report": str(arguments.rejection_report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
