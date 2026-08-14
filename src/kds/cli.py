from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds import __release__, __version__
from kds.audio.contracts import AudioLimits, AudioPipelineError, PreparationStatus
from kds.audio.pipeline import AudioPreparationPipeline
from kds.audio.windows import WindowConfig
from kds.data.assets import validate_assets
from kds.data.consents import (
    ConsentRegistryError,
    load_consent_registry,
    product_eligible_speaker_ids,
)
from kds.data.licenses import (
    LicenseLedgerError,
    TrainingProtocolError,
    load_license_ledger,
    validate_manifest_licenses,
    validate_training_protocol,
)
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest
from kds.data.source_matrix import (
    SourceMatrixError,
    load_source_mixed_research_matrix,
    validate_source_mixed_research_matrix,
)
from kds.data.split import GroupSplitter, SplitConfig
from kds.data.unseen_generator_ood import (
    UnseenGeneratorSuiteError,
    load_unseen_generator_suite,
    validate_unseen_generator_suite,
)

DEFAULT_RESEARCH_INFERENCE_CONTRACT = Path(
    "configs/inference/b0_user_audio_local_research_v1.json"
)


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


def _validate_training_protocol(arguments: argparse.Namespace) -> int:
    try:
        rows = load_manifest(Path(arguments.path))
        report = validate_training_protocol(
            rows,
            load_license_ledger(Path(arguments.license_ledger)),
            purpose=arguments.purpose,
        )
    except (LicenseLedgerError, ManifestError, TrainingProtocolError) as error:
        issues = list(error.issues)
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "purpose": report.purpose,
                "split_counts": report.split_counts,
                "source_ids": report.source_ids,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _validate_consent_registry(arguments: argparse.Namespace) -> int:
    try:
        entries = load_consent_registry(Path(arguments.path))
        eligible_speaker_ids = product_eligible_speaker_ids(entries)
    except ConsentRegistryError as error:
        print(json.dumps({"status": "error", "issues": list(error.issues)}, ensure_ascii=False))
        return 2
    active_count = sum(entry.status == "active" for entry in entries)
    print(
        json.dumps(
            {
                "status": "ok",
                "records": len(entries),
                "active_speakers": active_count,
                "product_eligible_speakers": len(eligible_speaker_ids),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _validate_source_mixed_research_matrix(arguments: argparse.Namespace) -> int:
    try:
        matrix = load_source_mixed_research_matrix(Path(arguments.path))
        report = validate_source_mixed_research_matrix(
            matrix, load_license_ledger(Path(arguments.license_ledger))
        )
    except (LicenseLedgerError, ManifestError, SourceMatrixError) as error:
        issues = list(error.issues)
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "protocol_id": report.protocol_id,
                "purpose": report.purpose,
                "source_ids": report.source_ids,
                "roles": [
                    {
                        "name": role.name,
                        "manifest": role.manifest_path,
                        "source_split": role.source_split,
                        "rows": role.rows,
                        "label_counts": role.label_counts,
                        "source_ids": role.source_ids,
                    }
                    for role in report.roles
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _validate_unseen_generator_suite(arguments: argparse.Namespace) -> int:
    try:
        suite = load_unseen_generator_suite(Path(arguments.path))
        report = validate_unseen_generator_suite(
            suite, load_license_ledger(Path(arguments.license_ledger))
        )
    except (LicenseLedgerError, ManifestError, UnseenGeneratorSuiteError) as error:
        print(json.dumps({"status": "error", "issues": list(error.issues)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "protocol_id": report.protocol_id,
                "purpose": report.purpose,
                "train": {"rows": report.train_rows, "source_ids": report.train_sources},
                "dev": {"rows": report.dev_rows, "source_ids": report.dev_sources},
                "train_dev_generator_families": report.train_dev_generator_families,
                "final_tests": [
                    {
                        "id": test.test_id,
                        "manifest": test.manifest_path,
                        "rows": test.rows,
                        "source_ids": test.source_ids,
                        "generator_families": test.generator_families,
                        "label_counts": test.label_counts,
                    }
                    for test in report.final_tests
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _assign_splits(arguments: argparse.Namespace) -> int:
    try:
        rows = load_manifest(Path(arguments.input_path))
        config = SplitConfig(
            train_ratio=float(arguments.train_ratio),
            dev_ratio=float(arguments.dev_ratio),
            test_ratio=float(arguments.test_ratio),
            seed=str(arguments.seed),
            preserve_ood=not bool(arguments.reassign_ood),
            include_voice_id=bool(arguments.include_voice_id),
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


def _validate_research_inference(arguments: argparse.Namespace) -> int:
    from kds.inference import ResearchInferenceContractError, load_research_inference_engine

    try:
        engine = load_research_inference_engine(Path(arguments.contract))
    except ResearchInferenceContractError as error:
        print(json.dumps({"status": "error", "issues": list(error.issues)}, ensure_ascii=False))
        return 2
    contract = engine.contract
    print(
        json.dumps(
            {
                "status": "ready",
                "research_only": True,
                "contract_id": contract.contract_id,
                "contract_sha256": contract.sha256,
                "model_version": engine.model_version,
                "checkpoint_sha256": contract.checkpoint.sha256,
                "calibrated": contract.calibrated,
                "probability_claim": contract.probability_claim,
                "fraud_claim": contract.fraud_claim,
                "product_grade": contract.product_grade,
                "warning": contract.warning,
                "limitations": list(contract.limitations),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _research_infer(arguments: argparse.Namespace) -> int:
    from kds.inference import (
        ResearchInferenceContractError,
        ResearchInferenceError,
        assert_user_audio_path_allowed,
        file_sha256,
        load_research_inference_engine,
    )

    if not arguments.acknowledge_research_only:
        from kds.inference import RESEARCH_ONLY_WARNING

        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "research_acknowledgement_required",
                    "warning": RESEARCH_ONLY_WARNING,
                },
                ensure_ascii=False,
            )
        )
        return 2
    try:
        engine = load_research_inference_engine(Path(arguments.contract))
        source = assert_user_audio_path_allowed(engine.contract, Path(arguments.path))
        input_sha256 = file_sha256(source)
        preprocessing = engine.contract.preprocessing
        pipeline = AudioPreparationPipeline(
            limits=AudioLimits(
                target_sample_rate=preprocessing.target_sample_rate,
                minimum_speech_seconds=preprocessing.minimum_speech_seconds,
            ),
            window_config=WindowConfig(
                samples=preprocessing.window_samples,
                hop_samples=preprocessing.hop_samples,
            ),
        )
        prepared = pipeline.prepare(source, arguments.mime_type)
        result = engine.score(prepared) if prepared.status is PreparationStatus.READY else None
    except AudioPipelineError as error:
        print(
            json.dumps(
                {"status": "error", "code": error.code.value, "detail": error.detail},
                ensure_ascii=False,
            )
        )
        return 2
    except (ResearchInferenceContractError, ResearchInferenceError) as error:
        issues = (
            list(error.issues)
            if isinstance(error, ResearchInferenceContractError)
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    contract = engine.contract
    payload: dict[str, object] = {
        "status": "ok" if result is not None else prepared.status.value,
        "research_only": True,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.sha256,
        "model_version": engine.model_version,
        "input_sha256": input_sha256,
        "speech_seconds": round(prepared.speech_seconds, 6),
        "quality_flags": list(prepared.quality_flags),
        "raw_spoof_logit": None,
        "uncalibrated_spoof_score": None,
        "interpretation": None,
        "calibrated": contract.calibrated,
        "probability_claim": contract.probability_claim,
        "fraud_claim": contract.fraud_claim,
        "product_grade": contract.product_grade,
        "warning": contract.warning,
        "limitations": list(contract.limitations),
        "windows": [],
    }
    if result is not None:
        payload.update(
            {
                "raw_spoof_logit": result.raw_spoof_logit,
                "uncalibrated_spoof_score": result.uncalibrated_spoof_score,
                "interpretation": result.interpretation,
                "windows": [
                    {
                        "start_s": window.start_s,
                        "end_s": window.end_s,
                        "real_samples": window.real_samples,
                        "raw_spoof_logit": window.raw_spoof_logit,
                        "uncalibrated_spoof_score": window.uncalibrated_spoof_score,
                        "interpretation": window.interpretation,
                    }
                    for window in result.windows
                ],
            }
        )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kds", description="KazDeepScan audited personal-research data and audio utilities"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__} ({__release__})"
    )
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

    protocol = subparsers.add_parser(
        "validate-training-protocol",
        help="Check explicit research/product eligibility before binary model training",
    )
    protocol.add_argument("path", help="Full binary protocol manifest CSV")
    protocol.add_argument("--license-ledger", required=True)
    protocol.add_argument("--purpose", choices=("research", "product"), required=True)
    protocol.set_defaults(handler=_validate_training_protocol)

    consent_registry = subparsers.add_parser(
        "validate-consent-registry",
        help="Validate a local pseudonymous registry for consented product-corpus speakers",
    )
    consent_registry.add_argument("path", help="Local CSV; must never be committed to Git")
    consent_registry.set_defaults(handler=_validate_consent_registry)

    source_matrix = subparsers.add_parser(
        "validate-source-matrix",
        help="Validate an explicit source-disjoint personal-research train/dev/test matrix",
    )
    source_matrix.add_argument("path", help="Versioned JSON source matrix")
    source_matrix.add_argument("--license-ledger", required=True)
    source_matrix.set_defaults(handler=_validate_source_mixed_research_matrix)

    unseen_suite = subparsers.add_parser(
        "validate-unseen-generator-suite",
        help=(
            "Validate frozen, source-safe final tests whose spoof families are unseen in train/dev"
        ),
    )
    unseen_suite.add_argument("path", help="Versioned JSON unseen-generator suite")
    unseen_suite.add_argument("--license-ledger", required=True)
    unseen_suite.set_defaults(handler=_validate_unseen_generator_suite)

    assign = subparsers.add_parser(
        "assign-splits", help="Assign train/dev/test by connected group, speaker, and text"
    )
    assign.add_argument("input_path", help="Validated input manifest CSV")
    assign.add_argument("output_path", help="New manifest path; must not already exist")
    assign.add_argument("--train-ratio", type=float, default=0.8)
    assign.add_argument("--dev-ratio", type=float, default=0.1)
    assign.add_argument("--test-ratio", type=float, default=0.1)
    assign.add_argument("--seed", default="20260808")
    assign.add_argument(
        "--include-voice-id",
        action="store_true",
        help=(
            "Keep a non-empty spoof voice_id in one split; required for a verified product "
            "voice group."
        ),
    )
    assign.add_argument("--reassign-ood", action="store_true")
    assign.set_defaults(handler=_assign_splits)

    validate_research = subparsers.add_parser(
        "validate-research-inference",
        help="Verify the separate research-only user-audio contract and local checkpoint",
    )
    validate_research.add_argument(
        "--contract",
        default=str(DEFAULT_RESEARCH_INFERENCE_CONTRACT),
        help="Strict versioned user-inference contract JSON",
    )
    validate_research.set_defaults(handler=_validate_research_inference)

    research_infer = subparsers.add_parser(
        "research-infer",
        help="Score one external user audio file with an uncalibrated research-only model",
    )
    research_infer.add_argument("path", help="External user audio; project data/model roots fail")
    research_infer.add_argument("--mime-type", default=None)
    research_infer.add_argument(
        "--contract",
        default=str(DEFAULT_RESEARCH_INFERENCE_CONTRACT),
        help="Strict versioned user-inference contract JSON",
    )
    research_infer.add_argument(
        "--acknowledge-research-only",
        action="store_true",
        help="Confirm that the uncalibrated result is not fraud proof or a product decision",
    )
    research_infer.set_defaults(handler=_research_infer)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))
