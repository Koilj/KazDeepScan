"""Run the one permitted offline, non-candidate, text-only VoxCPM2 CUDA smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import sha256_file
from kds.data.voxcpm2 import (
    VOXCPM2_MODEL_REVISION,
    VOXCPM2_SOURCE_REVISION,
    audit_voxcpm2_artifacts,
)
from kds.data.voxcpm2_text_only import (
    VOXCPM2_FIXED_SEED,
    bind_text,
    local_model_load_kwargs,
    offline_environment,
    synthesize_text_only,
)
from kds.eval.voxcpm2_smoke import (
    OneCallModel,
    VoxCPM2SmokeError,
    audit_waveform,
    install_python_network_guard,
    installed_distribution_audit,
    screen_smoke_text_against_denis,
)

SMOKE_TEXT = "Это отдельная техническая проверка локального синтеза."
EXPECTED_UV_LOCK_SHA256 = "fc066d21d09656c5060892baad096c53af6774c0947fad5bf6c676ea73c47c9b"
EXPECTED_WRAPPER_SHA256 = "3dcc290594a6af2670203b1dfd9ff500b96dbaf425b5ebe21011abfe57f12cbd"
EXPECTED_KEY_DISTRIBUTIONS = {
    "torch": "2.10.0",
    "torchaudio": "2.10.0",
    "torchcodec": "0.10.0",
    "transformers": "5.3.0",
    "voxcpm": "2.0.3.post23+gee8161e9e",
}


def _git_output(source_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise VoxCPM2SmokeError(
            f"Cannot verify runtime source checkout: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _require_runtime(source_root: Path, project_root: Path) -> dict[str, object]:
    if sys.version_info[:2] != (3, 12):
        raise VoxCPM2SmokeError(f"Smoke requires isolated CPython 3.12, got {sys.version}.")
    if os.environ.get("KDS_NETWORK_NAMESPACE") != "bwrap_unshare_net":
        raise VoxCPM2SmokeError("Smoke must be launched inside bwrap --unshare-net.")
    if _git_output(source_root, "rev-parse", "HEAD") != VOXCPM2_SOURCE_REVISION:
        raise VoxCPM2SmokeError("Runtime source checkout is not the pinned commit.")
    if _git_output(source_root, "status", "--short"):
        raise VoxCPM2SmokeError("Runtime source checkout is dirty.")
    uv_lock = source_root / "uv.lock"
    if sha256_file(uv_lock) != EXPECTED_UV_LOCK_SHA256:
        raise VoxCPM2SmokeError("Runtime uv.lock hash mismatch.")
    wrapper = project_root / "src/kds/data/voxcpm2_text_only.py"
    if sha256_file(wrapper) != EXPECTED_WRAPPER_SHA256:
        raise VoxCPM2SmokeError("Frozen text-only wrapper hash mismatch.")
    distributions = installed_distribution_audit()
    versions = dict(distributions.distributions)
    for name, expected in EXPECTED_KEY_DISTRIBUTIONS.items():
        if versions.get(name) != expected:
            raise VoxCPM2SmokeError(
                f"Installed {name} version mismatch: expected {expected}, got {versions.get(name)}."
            )
    return {
        "python_version": sys.version,
        "uv_lock_sha256": EXPECTED_UV_LOCK_SHA256,
        "installed_distributions": [
            {"name": name, "version": version}
            for name, version in distributions.distributions
        ],
        "installed_distribution_count": len(distributions.distributions),
        "installed_distribution_fingerprint": distributions.fingerprint,
        "key_distributions": EXPECTED_KEY_DISTRIBUTIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--denis-archive", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    partial = arguments.output_wav.with_suffix(".partial.wav")
    try:
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if arguments.output_wav.exists() or partial.exists() or arguments.output_receipt.exists():
            raise VoxCPM2SmokeError("Smoke WAV/partial/receipt output must be new.")
        if not arguments.output_wav.parent.is_dir() or not arguments.output_receipt.parent.is_dir():
            raise VoxCPM2SmokeError("Smoke output parent directories must already exist.")
        project_root = arguments.project_root.resolve(strict=True)
        source_root = arguments.runtime_source_root.resolve(strict=True)
        runtime = _require_runtime(source_root, project_root)
        for name, value in offline_environment().items():
            os.environ[name] = value
        network_attempts = install_python_network_guard()

        artifact_audit = audit_voxcpm2_artifacts(
            arguments.model_root.resolve(strict=True),
            arguments.source_archive.resolve(strict=True),
        )
        non_candidate = screen_smoke_text_against_denis(
            arguments.denis_archive.resolve(strict=True), SMOKE_TEXT
        )

        import torch
        from voxcpm import VoxCPM  # type: ignore[import-not-found]

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise VoxCPM2SmokeError("Pinned runtime has no CUDA/BF16 device.")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        load_kwargs = local_model_load_kwargs(arguments.model_root)
        model = VoxCPM.from_pretrained(**load_kwargs)
        load_seconds = time.monotonic() - started
        audited_model = OneCallModel(model)
        binding = bind_text(SMOKE_TEXT)
        generation_started = time.monotonic()
        waveform_value = synthesize_text_only(audited_model, SMOKE_TEXT, binding)
        generation_seconds = time.monotonic() - generation_started
        waveform = np.asarray(waveform_value, dtype=np.float32)
        waveform_audit = audit_waveform(waveform, artifact_audit.output_sample_rate_hz)
        sf.write(partial, waveform, artifact_audit.output_sample_rate_hz, subtype="PCM_16")
        stored, stored_rate = sf.read(partial, dtype="float32", always_2d=True)
        if stored_rate != 48_000 or stored.shape != (waveform_audit.frames, 1):
            raise VoxCPM2SmokeError("Stored smoke WAV shape/sample-rate mismatch.")
        partial.replace(arguments.output_wav)
        if audited_model.calls != 1 or audited_model.kwargs is None:
            raise VoxCPM2SmokeError("Smoke did not execute exactly one generation call.")
        if network_attempts:
            raise VoxCPM2SmokeError(
                f"Upstream attempted network access under the guard: {network_attempts}"
            )

        generation_kwargs = audited_model.kwargs
        receipt = {
            "schema_version": 1,
            "protocol_id": "voxcpm2-official-text-only-cuda-smoke-v1",
            "created_at": arguments.created_at,
            "status": "passed_non_candidate_technical_smoke",
            "model_revision": VOXCPM2_MODEL_REVISION,
            "source_revision": VOXCPM2_SOURCE_REVISION,
            "artifact_source_receipt": {
                "path": "data/licenses/voxcpm2_official_v1_artifact_source_receipt.json",
                "sha256": sha256_file(
                    project_root
                    / "data/licenses/voxcpm2_official_v1_artifact_source_receipt.json"
                ),
            },
            "runtime": runtime,
            "network_policy": {
                "outer_namespace": "bwrap --unshare-net",
                "offline_environment": dict(offline_environment()),
                "python_socket_guard_installed_before_upstream_import": True,
                "observed_upstream_network_attempts": len(network_attempts),
            },
            "non_candidate_text_screen": non_candidate,
            "model_load": {
                "kwargs": load_kwargs,
                "seconds": f"{load_seconds:.6f}",
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_runtime": torch.version.cuda,
                "torch_version": torch.__version__,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            },
            "generation": {
                "call_count": audited_model.calls,
                "kwargs": generation_kwargs,
                "seconds": f"{generation_seconds:.6f}",
                "fixed_seed": VOXCPM2_FIXED_SEED,
                "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "output": {
                "local_path_policy": "ignored models/ runtime directory; not stored in Git",
                "wav_sha256": sha256_file(arguments.output_wav),
                "wav_size_bytes": arguments.output_wav.stat().st_size,
                "container": "WAV",
                "subtype": "PCM_16",
                **asdict(waveform_audit),
            },
            "smoke_runner_sha256": sha256_file(Path(__file__)),
            "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
            "claims": {
                "local_model_load_verified": True,
                "text_only_default_voice_smoke_verified": True,
                "reference_or_prompt_audio_used": False,
                "voice_cloning_used": False,
                "semantic_normalizer_used": False,
                "denoiser_used": False,
                "retry_or_resynthesis_used": False,
                "candidate_text_used": False,
                "detector_inference_performed": False,
                "candidate_selection_or_synthesis_authorized": False,
                "training_data_overlap": "unverified",
                "default_voice_identity": "unknown_not_claimed",
            },
            "limitations": [
                "This is one non-candidate technical smoke, not an acoustic/language review.",
                "It does not authorize candidate selection, synthesis, detector inference, "
                "or rerun.",
                "The upstream lock installs only in frozen mode with current uv because "
                "project metadata would otherwise request a lock update.",
            ],
        }
        with arguments.output_receipt.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
        partial.unlink(missing_ok=True)
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output_receipt": str(arguments.output_receipt),
                "receipt_sha256": sha256_file(arguments.output_receipt),
                "wav_sha256": sha256_file(arguments.output_wav),
                "frames": waveform_audit.frames,
                "generation_calls": audited_model.calls,
                "network_attempts": len(network_attempts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
