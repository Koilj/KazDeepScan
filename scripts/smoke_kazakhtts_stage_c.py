"""Run the frozen RU/KK/mixed acoustic smoke without detector inference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
import yaml  # type: ignore[import-untyped]
from packaging.version import Version

from kds.data.assets import sha256_file
from kds.data.kazakhtts import (
    KazakhTtsRuntime,
    extract_verified_kazakhtts_runtime,
    load_kazakhtts_runtime,
    validate_kazakhtts_text,
)
from kds.data.research_tts import (
    ResearchTtsError,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.eval.kazakhtts_smoke import KazakhTtsSmokeError, load_kazakhtts_smoke_plan


def _device(value: str) -> torch.device:
    resolved = ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ResearchTtsError("KazakhTTS CUDA was requested but is unavailable.")
    return device


def _disable_unused_english_g2p_download_route() -> None:
    """ESPnet imports g2p_en eagerly even though this char model declares g2p=null."""

    module = types.ModuleType("g2p_en")

    class DisabledG2p:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("g2p_en is disabled for the pinned KazakhTTS char model.")

    module.G2p = DisabledG2p  # type: ignore[attr-defined]
    sys.modules["g2p_en"] = module


def _load_models(
    runtime: KazakhTtsRuntime,
    extracted: Any,
    device: torch.device,
) -> tuple[Any, Any]:
    if Version(torch.__version__.split("+", maxsplit=1)[0]) < Version("2.6"):
        raise ResearchTtsError("KazakhTTS requires torch>=2.6 for weights-only load by default.")
    for distribution, expected in (
        ("espnet", runtime.espnet_version),
        ("parallel-wavegan", runtime.parallel_wavegan_version),
    ):
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise ResearchTtsError(
                f"KazakhTTS runtime needs {distribution}=={expected}, got {actual}."
            )
    _disable_unused_english_g2p_download_route()
    import scipy.signal as signal  # type: ignore[import-untyped]

    signal.kaiser = signal.windows.kaiser
    from espnet2.bin.tts_inference import Text2Speech  # type: ignore[import-untyped]
    from parallel_wavegan.utils import load_model  # type: ignore[import-untyped]

    previous_directory = Path.cwd()
    try:
        os.chdir(extracted.acoustic_config.parents[2])
        text_to_speech = Text2Speech(
            extracted.acoustic_config,
            extracted.acoustic_checkpoint,
            device=str(device),
            threshold=runtime.threshold,
            minlenratio=runtime.min_length_ratio,
            maxlenratio=runtime.max_length_ratio,
            use_att_constraint=runtime.use_attention_constraint,
            backward_window=runtime.backward_window,
            forward_window=runtime.forward_window,
            speed_control_alpha=runtime.speed_control_alpha,
        )
    finally:
        os.chdir(previous_directory)
    text_to_speech.spc2wav = None
    try:
        vocoder_config_value: object = yaml.safe_load(
            extracted.vocoder_config.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ResearchTtsError(f"Cannot safely read KazakhTTS vocoder config: {error}") from error
    if not isinstance(vocoder_config_value, dict):
        raise ResearchTtsError("KazakhTTS vocoder config must be a mapping.")
    vocoder = load_model(
        str(extracted.vocoder_checkpoint), config=cast(dict[str, object], vocoder_config_value)
    )
    vocoder = vocoder.to(device).eval()
    vocoder.remove_weight_norm()
    return text_to_speech, vocoder


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
        device = _device(arguments.device)
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
                text_to_speech, vocoder = _load_models(runtime, extracted, device)
                torch.manual_seed(plan.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(plan.seed)
                outputs: list[dict[str, object]] = []
                for case in plan.cases:
                    text = validate_kazakhtts_text(case.text, extracted)
                    with torch.inference_mode():
                        result = text_to_speech(text)
                        features = result["feat_gen"]
                        waveform = vocoder.inference(features).reshape(-1).float().cpu().numpy()
                    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
                        raise ResearchTtsError(
                            f"KazakhTTS smoke {case.case_id!r} produced invalid samples."
                        )
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
