"""Create a narrow, input-pinned KSC2 mixed-text Silero V4 research candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import require_valid_assets, sha256_file
from kds.data.ksc2_mixed_candidate import (
    Ksc2MixedCandidateError,
    load_published_mixed_review,
)
from kds.data.ksc2_mixed_silero_v4 import (
    KSC2_MIXED_SILERO_V4_SOURCE_ID,
    build_paired_mixed_candidate_rows,
    mixed_silero_v4_spoof_row,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, load_manifest, validate_manifest, write_manifest
from kds.data.preprocess import preprocess_rows
from kds.data.research_tts import (
    ResearchTtsError,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.data.silero_v4 import (
    SileroV4Error,
    load_silero_v4_model,
    load_silero_v4_runtime,
    normalize_silero_v4_text,
    synthesize_silero_v4,
)


def _safe_slice_name(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"} or (
        device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Requested Silero device is unavailable.")
    return device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a narrow input-pinned KSC2 mixed-text Silero candidate. It is research-only "
            "and does not certify generated acoustic language preservation."
        )
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--text-rejection-report", type=Path, required=True)
    parser.add_argument("--audio-rejection-report", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args(argv)

    try:
        outputs = (
            arguments.raw_manifest,
            arguments.ready_manifest,
            arguments.text_rejection_report,
            arguments.audio_rejection_report,
            arguments.candidate_manifest,
        )
        if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
            raise Ksc2MixedCandidateError(["Every mixed Silero output must be distinct and new."])
        if any(not path.parent.is_dir() for path in outputs):
            raise Ksc2MixedCandidateError(["Every mixed Silero output parent must already exist."])
        slice_name = _safe_slice_name(arguments.slice_name)
        data_root = arguments.data_root.resolve(strict=True)
        base_rows = load_manifest(arguments.base_manifest)
        validate_manifest(base_rows)
        evidence = load_published_mixed_review(arguments.review_csv, arguments.review_receipt)
        evidence_by_id = {item.annotation_id: item for item in evidence}
        if not base_rows or any(
            row.source_name != "ksc2_v1"
            or row.split != "test"
            or row.label != "bonafide"
            or row.language != "mixed"
            or row.code_switch != "true"
            or row.sample_id not in evidence_by_id
            or row.text_hash != evidence_by_id[row.sample_id].transcript_sha256
            for row in base_rows
        ):
            raise Ksc2MixedCandidateError(
                ["Base manifest must be a subset of the pinned ready KSC2 mixed evidence."]
            )
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        require_valid_assets(base_rows, data_root)
        if KSC2_MIXED_SILERO_V4_SOURCE_ID not in ledger:
            raise LicenseLedgerError(["KSC2 mixed Silero source is missing from license ledger."])
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("KSC2 mixed Silero lock must contain exactly one model.")
        model_spec = lock.models[0]
        runtime = load_silero_v4_runtime(model_spec)
        profile = runtime.profiles_by_language["kk"][0]
        text_rejections: list[dict[str, str]] = []
        eligible = []
        for row in base_rows:
            text = evidence_by_id[row.sample_id].transcript
            try:
                normalized = normalize_silero_v4_text(text)
            except SileroV4Error as error:
                text_rejections.append(
                    {"sample_id": row.sample_id, "text_hash": row.text_hash, "reason": str(error)}
                )
            else:
                eligible.append((row, normalized))
        if not eligible:
            raise Ksc2MixedCandidateError(["No KSC2 mixed transcript is Silero-compatible."])
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
        device = _device(arguments.device)
        output_directory = (
            data_root / "raw" / KSC2_MIXED_SILERO_V4_SOURCE_ID / "slices" / slice_name
        )
        if output_directory.exists():
            raise Ksc2MixedCandidateError(
                [f"Refusing to overwrite generated slice: {output_directory}"]
            )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-ksc2-mixed-silero-", dir=output_directory.parent)
        )
        stage_report_directory = Path(
            tempfile.mkdtemp(
                prefix="kds-ksc2-mixed-silero-report-", dir=arguments.raw_manifest.parent
            )
        )
        try:
            model = load_silero_v4_model(
                verified[model_spec.model_id][runtime.package_path], runtime, device
            )
            raw_rows = []
            for base_row, normalized in eligible:
                filename = f"{base_row.sample_id.encode().hex()[-16:]}.wav"
                relative_path = (
                    Path("raw")
                    / KSC2_MIXED_SILERO_V4_SOURCE_ID
                    / "slices"
                    / slice_name
                    / model_spec.model_id
                    / filename
                )
                output = stage_directory / model_spec.model_id / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                synthesize_silero_v4(
                    model=model,
                    profile=profile,
                    text=normalized,
                    runtime=runtime,
                    output=output,
                )
                info = sf.info(str(output))
                if info.duration <= 0 or info.samplerate != runtime.sample_rate:
                    raise RuntimeError(f"Silero output is invalid for {base_row.sample_id}.")
                raw_rows.append(
                    mixed_silero_v4_spoof_row(
                        base_row=base_row,
                        model=model_spec,
                        profile=profile,
                        relative_path=relative_path.as_posix(),
                        sha256=sha256_file(output),
                        duration_s=float(info.duration),
                        original_sr=int(info.samplerate),
                        created_at=arguments.created_at,
                        device=f"local_{device.type}_silero_v4_fastpitch_hifigan",
                    )
                )
            validate_manifest(raw_rows)
            validate_manifest_licenses(raw_rows, ledger)
            stage_raw_manifest = stage_report_directory / arguments.raw_manifest.name
            stage_text_report = stage_report_directory / arguments.text_rejection_report.name
            write_manifest(stage_raw_manifest, raw_rows)
            stage_text_report.write_text(
                json.dumps(
                    {
                        "base_manifest": arguments.base_manifest.as_posix(),
                        "base_manifest_sha256": sha256_file(arguments.base_manifest),
                        "review_csv_sha256": sha256_file(arguments.review_csv),
                        "model_lock_sha256": sha256_file(arguments.model_lock),
                        "published_rows": len(raw_rows),
                        "rejected_rows": text_rejections,
                        "rule": (
                            "Input-pinned mixed transcript only; "
                            "acoustic preservation is unverified."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if (
                output_directory.exists()
                or arguments.raw_manifest.exists()
                or arguments.text_rejection_report.exists()
            ):
                raise Ksc2MixedCandidateError(["Raw output appeared while staging."])
            stage_directory.replace(output_directory)
            stage_raw_manifest.replace(arguments.raw_manifest)
            stage_text_report.replace(arguments.text_rejection_report)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
            shutil.rmtree(stage_report_directory, ignore_errors=True)

        prepared = preprocess_rows(
            raw_rows, data_root, AudioPreparationPipeline(), allow_rejections=True
        )
        validate_manifest(prepared.processed_rows)
        validate_manifest_licenses(prepared.processed_rows, ledger)
        require_valid_assets(prepared.processed_rows, data_root)
        audio_rejected_ids = {issue.sample_id for issue in prepared.issues}
        candidate_rows = build_paired_mixed_candidate_rows(
            base_rows=base_rows,
            raw_spoof_rows=raw_rows,
            ready_spoof_rows=prepared.processed_rows,
            text_rejected_base_ids={item["sample_id"] for item in text_rejections},
            audio_rejected_spoof_ids=audio_rejected_ids,
        )
        validate_manifest(candidate_rows)
        validate_manifest_licenses(candidate_rows, ledger)
        require_valid_assets(candidate_rows, data_root)
        with tempfile.TemporaryDirectory(
            prefix="kds-ksc2-mixed-silero-ready-", dir=arguments.ready_manifest.parent
        ) as stage_name:
            stage = Path(stage_name)
            stage_ready = stage / arguments.ready_manifest.name
            stage_audio_report = stage / arguments.audio_rejection_report.name
            stage_candidate = stage / arguments.candidate_manifest.name
            write_manifest(stage_ready, prepared.processed_rows)
            stage_audio_report.write_text(
                json.dumps(
                    {
                        "raw_manifest": arguments.raw_manifest.as_posix(),
                        "raw_manifest_sha256": sha256_file(arguments.raw_manifest),
                        "published_rows": len(prepared.processed_rows),
                        "rejected_rows": [
                            {
                                "sample_id": item.sample_id,
                                "relative_path": item.relative_path,
                                "reason": item.detail,
                            }
                            for item in prepared.issues
                        ],
                        "rule": "Signal QA/VAD only; not an acoustic language-preservation check.",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            write_manifest(stage_candidate, candidate_rows)
            if any(
                path.exists()
                for path in (
                    arguments.ready_manifest,
                    arguments.audio_rejection_report,
                    arguments.candidate_manifest,
                )
            ):
                raise Ksc2MixedCandidateError(["Ready output appeared while staging."])
            stage_ready.replace(arguments.ready_manifest)
            stage_audio_report.replace(arguments.audio_rejection_report)
            stage_candidate.replace(arguments.candidate_manifest)
    except (
        Ksc2MixedCandidateError,
        LicenseLedgerError,
        ManifestError,
        ResearchTtsError,
        SileroV4Error,
        OSError,
        RuntimeError,
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
                "raw_spoof_rows": len(raw_rows),
                "ready_spoof_rows": len(prepared.processed_rows),
                "candidate_rows": len(candidate_rows),
                "text_rejections": len(text_rejections),
                "audio_rejections": len(prepared.issues),
                "candidate_manifest": str(arguments.candidate_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
