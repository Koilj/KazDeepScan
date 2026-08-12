"""Run the frozen RU/KK/mixed acoustic smoke without detector inference."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.assets import sha256_file
from kds.data.kazakhtts import (
    extract_verified_kazakhtts_runtime,
    load_kazakhtts_runtime,
    validate_kazakhtts_text,
)
from kds.data.kazakhtts_inference import (
    load_kazakhtts_models,
    resolve_kazakhtts_device,
    synthesize_kazakhtts_waveform,
)
from kds.data.research_tts import (
    ResearchTtsError,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.eval.kazakhtts_smoke import KazakhTtsSmokeError, load_kazakhtts_smoke_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the frozen pre-detector KazakhTTS Stage-C smoke packet."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    try:
        if arguments.output_directory.exists() or arguments.output_report.exists():
            raise ResearchTtsError("Refusing to overwrite KazakhTTS smoke outputs.")
        if (
            not arguments.output_directory.parent.is_dir()
            or not arguments.output_report.parent.is_dir()
        ):
            raise ResearchTtsError("KazakhTTS smoke output parent does not exist.")
        plan = load_kazakhtts_smoke_plan(arguments.plan)
        route_receipt = json.loads(plan.generator_route_gate.path.read_text(encoding="utf-8"))
        audit = route_receipt.get("audit", {})
        if (
            not isinstance(audit, dict)
            or audit.get("novelty_claim") != "unseen_exact_generator_route"
            or audit.get("exact_route_overlap_rows") != 0
        ):
            raise ResearchTtsError("KazakhTTS generator-route receipt did not pass its gate.")
        lock = load_research_tts_model_lock(plan.model_lock.path)
        if len(lock.models) != 1:
            raise ResearchTtsError("KazakhTTS smoke model lock must contain exactly one model.")
        model = lock.models[0]
        runtime = load_kazakhtts_runtime(model)
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
        device = resolve_kazakhtts_device(arguments.device)
        stage_directory = Path(
            tempfile.mkdtemp(
                prefix=".kds-kazakhtts-smoke-", dir=arguments.output_directory.parent
            )
        )
        staged_report_directory = Path(
            tempfile.mkdtemp(
                prefix=".kds-kazakhtts-report-", dir=arguments.output_report.parent
            )
        )
        try:
            with tempfile.TemporaryDirectory(prefix="kds-kazakhtts-runtime-") as runtime_dir:
                extracted = extract_verified_kazakhtts_runtime(
                    verified_paths=verified[model.model_id],
                    runtime=runtime,
                    destination=Path(runtime_dir) / "runtime",
                )
                text_to_speech, vocoder = load_kazakhtts_models(runtime, extracted, device)
                torch.manual_seed(plan.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(plan.seed)
                outputs: list[dict[str, object]] = []
                for case in plan.cases:
                    text = validate_kazakhtts_text(case.text, extracted)
                    waveform = synthesize_kazakhtts_waveform(text_to_speech, vocoder, text)
                    output = stage_directory / f"{case.case_id}.wav"
                    sf.write(output, waveform, runtime.sample_rate, subtype="PCM_16")
                    info = sf.info(str(output))
                    if (
                        info.samplerate != runtime.sample_rate
                        or info.channels != 1
                        or str(info.format).lower() != "wav"
                        or not math.isfinite(info.duration)
                        or info.duration <= 0
                    ):
                        raise ResearchTtsError(
                            f"KazakhTTS smoke {case.case_id!r} produced an invalid WAV."
                        )
                    outputs.append(
                        {
                            "case_id": case.case_id,
                            "language": case.language,
                            "support_status": case.status,
                            "text": case.text,
                            "relative_path": (
                                arguments.output_directory / output.name
                            ).as_posix(),
                            "sha256": sha256_file(output),
                            "size_bytes": output.stat().st_size,
                            "duration_s": info.duration,
                            "sample_rate": info.samplerate,
                            "channels": info.channels,
                            "peak_abs": float(np.max(np.abs(waveform))),
                            "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
                        }
                    )
            report = {
                "schema_version": 1,
                "protocol_id": plan.protocol_id,
                "created_at": arguments.created_at,
                "plan": {
                    "path": arguments.plan.as_posix(),
                    "sha256": sha256_file(arguments.plan),
                },
                "model_lock_sha256": plan.model_lock.sha256,
                "generator_route_gate_sha256": plan.generator_route_gate.sha256,
                "model_id": model.model_id,
                "generator_family": model.generator_family,
                "fixed_voice_id": runtime.fixed_voice_id,
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "detector_inference_performed": False,
                "technical_smoke_passed": True,
                "acoustic_language_gate_passed": False,
                "outputs": outputs,
                "interpretation": (
                    "All files are technical smoke only. KK may proceed to human acoustic "
                    "review; RU/mixed remain conditional and unsupported until that review."
                ),
            }
            staged_report = staged_report_directory / arguments.output_report.name
            staged_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if arguments.output_directory.exists() or arguments.output_report.exists():
                raise ResearchTtsError("KazakhTTS smoke output appeared during staging.")
            stage_directory.replace(arguments.output_directory)
            staged_report.replace(arguments.output_report)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
            shutil.rmtree(staged_report_directory, ignore_errors=True)
    except (
        ImportError,
        KazakhTtsSmokeError,
        OSError,
        ResearchTtsError,
        RuntimeError,
        ValueError,
    ) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output_directory": str(arguments.output_directory),
                "output_report": str(arguments.output_report),
                "cases": len(outputs),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
