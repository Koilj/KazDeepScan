from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.audio.contracts import AudioPipelineError
from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import validate_assets
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest
from kds.data.split import GroupSplitter, SplitConfig


def _inspect_audio(arguments: argparse.Namespace) -> int:
    try:
        prepared = AudioPreparationPipeline().prepare(Path(arguments.path), arguments.mime_type)
    except AudioPipelineError as error:
        print(json.dumps({"status": "error", "code": error.code.value, "detail": error.detail}))
        return 2
    payload = {
        "status": prepared.status.value,
        "duration_seconds": round(prepared.media.duration_seconds, 3),
        "speech_seconds": round(prepared.speech_seconds, 3),
        "quality": {
            "peak": round(prepared.quality.peak, 6),
            "rms_dbfs": round(prepared.quality.rms_dbfs, 3),
            "clipped_fraction": round(prepared.quality.clipped_fraction, 6),
            "dc_offset": round(prepared.quality.dc_offset, 6),
        },
        "quality_flags": list(prepared.quality_flags),
        "speech_segments": [
            {"start_s": round(segment.start_seconds, 3), "end_s": round(segment.end_seconds, 3)}
            for segment in prepared.speech_segments
        ],
        "windows": [
            {
                "start_s": round(window.start_seconds, 3),
                "end_s": round(window.end_seconds, 3),
                "real_samples": window.real_samples,
            }
            for window in prepared.windows
        ],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _validate_manifest(arguments: argparse.Namespace) -> int:
    try:
        rows = load_manifest(Path(arguments.path))
        validate_manifest(rows, require_ood_generator=arguments.require_ood_generator)
        if arguments.license_ledger is not None:
            validate_manifest_licenses(rows, load_license_ledger(Path(arguments.license_ledger)))
    except (LicenseLedgerError, ManifestError) as error:
        print(json.dumps({"status": "error", "issues": list(error.issues)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", "rows": len(rows)}, ensure_ascii=False))
    return 0


def _validate_assets(arguments: argparse.Namespace) -> int:
    try:
        rows = load_manifest(Path(arguments.path))
        report = validate_assets(
            rows,
            Path(arguments.audio_root),
            verify_sha256=not bool(arguments.skip_sha256),
        )
    except ManifestError as error:
        print(json.dumps({"status": "error", "issues": list(error.issues)}, ensure_ascii=False))
        return 2
    payload = {
        "status": "ok" if report.is_valid else "error",
        "checked": report.checked,
        "verified": report.verified,
        "issues": [
            {
                "sample_id": issue.sample_id,
                "relative_path": issue.relative_path,
                "detail": issue.detail,
            }
            for issue in report.issues
        ],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if report.is_valid else 2


def _assign_splits(arguments: argparse.Namespace) -> int:
    try:
        rows = load_manifest(Path(arguments.input_path))
        config = SplitConfig(
            train_ratio=float(arguments.train_ratio),
            dev_ratio=float(arguments.dev_ratio),
            test_ratio=float(arguments.test_ratio),
            seed=str(arguments.seed),
            preserve_ood=not bool(arguments.reassign_ood),
        )
        assigned = GroupSplitter(config).assign_rows(rows)
        write_manifest(Path(arguments.output_path), assigned)
    except (ManifestError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    split_names = ("train", "dev", "test", "ood")
    counts = {split: sum(row.split == split for row in assigned) for split in split_names}
    print(json.dumps({"status": "ok", "rows": len(assigned), "split_counts": counts}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kds", description="KazDeepScan data and audio utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_audio = subparsers.add_parser(
        "inspect-audio", help="Validate and prepare one local audio file"
    )
    inspect_audio.add_argument("path")
    inspect_audio.add_argument("--mime-type", default=None)
    inspect_audio.set_defaults(handler=_inspect_audio)

    validate = subparsers.add_parser(
        "validate-manifest", help="Check manifest schema and split leakage"
    )
    validate.add_argument("path")
    validate.add_argument("--require-ood-generator", action="store_true")
    validate.add_argument(
        "--license-ledger",
        help="Require every manifest source to have an approved status in this CSV ledger.",
    )
    validate.set_defaults(handler=_validate_manifest)

    assets = subparsers.add_parser("validate-assets", help="Check audio assets and SHA-256 digests")
    assets.add_argument("path", help="Input manifest CSV")
    assets.add_argument("--audio-root", required=True)
    assets.add_argument("--skip-sha256", action="store_true")
    assets.set_defaults(handler=_validate_assets)

    assign = subparsers.add_parser(
        "assign-splits", help="Assign train/dev/test by connected group, speaker, and text"
    )
    assign.add_argument("input_path", help="Validated input manifest CSV")
    assign.add_argument("output_path", help="New manifest path; must not already exist")
    assign.add_argument("--train-ratio", type=float, default=0.8)
    assign.add_argument("--dev-ratio", type=float, default=0.1)
    assign.add_argument("--test-ratio", type=float, default=0.1)
    assign.add_argument("--seed", default="20260808")
    assign.add_argument("--reassign-ood", action="store_true")
    assign.set_defaults(handler=_assign_splits)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))
