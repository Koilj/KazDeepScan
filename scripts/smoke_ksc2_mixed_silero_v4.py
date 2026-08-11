"""Run a technical, non-dataset Silero V4 smoke test on KSC2 mixed transcripts."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import sha256_file
from kds.data.ksc2_mixed_candidate import (
    Ksc2MixedCandidateError,
    load_published_mixed_review,
    select_mixed_smoke_evidence,
)
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


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Silero smoke-test device must be CPU or CUDA.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Silero smoke-test requested CUDA but it is unavailable.")
    return device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tiny technical Silero V4 smoke set from hash-pinned KSC2 mixed review "
            "transcripts. It does not publish spoof labels, a manifest, or a quality claim."
        )
    )
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", default="20260811")
    parser.add_argument("--profile-language", choices=("ru", "kk"), action="append")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args(argv)

    try:
        profiles_requested = tuple(arguments.profile_language or ("ru", "kk"))
        if len(set(profiles_requested)) != len(profiles_requested):
            raise Ksc2MixedCandidateError(["Smoke-test profile languages must not repeat."])
        if arguments.output_directory.exists() or arguments.report.exists():
            raise Ksc2MixedCandidateError(["Refusing to overwrite a Silero smoke-test output."])
        if not arguments.output_directory.parent.is_dir() or not arguments.report.parent.is_dir():
            raise Ksc2MixedCandidateError(["Smoke-test output parents must already exist."])
        evidence = load_published_mixed_review(arguments.review_csv, arguments.review_receipt)
        selected = select_mixed_smoke_evidence(evidence, limit=arguments.limit, seed=arguments.seed)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Silero smoke-test lock must contain exactly one model.")
        model_spec = lock.models[0]
        runtime = load_silero_v4_runtime(model_spec)
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
        device = _device(arguments.device)
        profiles = tuple(
            runtime.profiles_by_language[language][0] for language in profiles_requested
        )
        stage_directory = Path(
            tempfile.mkdtemp(
                prefix="kds-ksc2-mixed-silero-smoke-", dir=arguments.output_directory.parent
            )
        )
        stage_report_directory = Path(
            tempfile.mkdtemp(prefix="kds-ksc2-mixed-silero-report-", dir=arguments.report.parent)
        )
        try:
            model = load_silero_v4_model(
                verified[model_spec.model_id][runtime.package_path], runtime, device
            )
            technical_rows: list[dict[str, object]] = []
            pipeline = AudioPreparationPipeline()
            for item in selected:
                normalized = normalize_silero_v4_text(item.transcript)
                for profile in profiles:
                    filename = f"{item.audio_sha256[:16]}-{profile.voice_id}.wav"
                    output = stage_directory / filename
                    synthesize_silero_v4(
                        model=model,
                        profile=profile,
                        text=normalized,
                        runtime=runtime,
                        output=output,
                    )
                    info = sf.info(str(output))
                    prepared = pipeline.prepare(output)
                    technical_rows.append(
                        {
                            "annotation_id": item.annotation_id,
                            "component": item.component,
                            "source_transcript_sha256": item.transcript_sha256,
                            "source_audio_sha256": item.audio_sha256,
                            "profile_language": profile.language,
                            "profile_id": profile.voice_id,
                            "output_file": filename,
                            "output_sha256": sha256_file(output),
                            "duration_s": float(info.duration),
                            "sample_rate": int(info.samplerate),
                            "technical_status": prepared.status.value,
                            "speech_seconds": prepared.speech_seconds,
                            "quality_flags": list(prepared.quality_flags),
                        }
                    )
            stage_report = stage_report_directory / arguments.report.name
            stage_report.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "technical_tts_smoke_only",
                        "review_csv": arguments.review_csv.as_posix(),
                        "review_csv_sha256": sha256_file(arguments.review_csv),
                        "review_receipt": arguments.review_receipt.as_posix(),
                        "review_receipt_sha256": sha256_file(arguments.review_receipt),
                        "model_lock": arguments.model_lock.as_posix(),
                        "model_lock_sha256": sha256_file(arguments.model_lock),
                        "verified_model_id": model_spec.model_id,
                        "device": f"local_{device.type}",
                        "selection_seed": arguments.seed,
                        "selected_review_rows": len(selected),
                        "generated_outputs": len(technical_rows),
                        "outputs": technical_rows,
                        "rule": (
                            "Technical synthesis and signal QA only. "
                            "No listening, intelligibility, "
                            "language-preservation, spoof, or benchmark claim is made."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if arguments.output_directory.exists() or arguments.report.exists():
                raise Ksc2MixedCandidateError(["A smoke-test output appeared while staging."])
            stage_directory.replace(arguments.output_directory)
            stage_report.replace(arguments.report)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
            shutil.rmtree(stage_report_directory, ignore_errors=True)
    except (
        Ksc2MixedCandidateError,
        ResearchTtsError,
        SileroV4Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        issues = list(error.issues) if isinstance(error, Ksc2MixedCandidateError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    ready_outputs = sum(row["technical_status"] == "ready" for row in technical_rows)
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_review_rows": len(selected),
                "generated_outputs": len(technical_rows),
                "technically_ready_outputs": ready_outputs,
                "output_directory": str(arguments.output_directory),
                "report": str(arguments.report),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
