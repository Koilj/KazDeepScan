from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, cast

import soundfile as sf  # type: ignore[import-untyped]
import torch
from transformers import AutoTokenizer, VitsModel

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.ksc_derived_kk import (
    KSC_DERIVED_KK_SOURCE_ID,
    assign_synthesis_profiles,
    derived_spoof_row,
    load_verified_ksc_transcript,
    select_ksc_bonafide_rows,
    synthesis_profiles,
    synthesis_seed,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.research_tts import (
    ResearchTtsError,
    ResearchTtsModel,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)


def _safe_slice_name(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    return value


def _mms_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("MMS CUDA was requested but is unavailable.")
    return device


def _runtime_path(paths: dict[str, Path], model: ResearchTtsModel, key: str) -> Path:
    value = model.runtime.get(key)
    if not isinstance(value, str) or value not in paths:
        raise ResearchTtsError(
            f"Model {model.model_id!r} runtime {key!r} is not a locked artifact."
        )
    return paths[value]


def _synthesize_piper(
    binary: str,
    model: ResearchTtsModel,
    paths: dict[str, Path],
    speaker_id: int,
    text: str,
    output: Path,
) -> None:
    model_path = _runtime_path(paths, model, "model_path")
    config_path = _runtime_path(paths, model, "config_path")
    try:
        result = subprocess.run(
            [
                binary,
                "-m",
                str(model_path),
                "-c",
                str(config_path),
                "-s",
                str(speaker_id),
                "-f",
                str(output),
            ],
            input=text,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Piper synthesis failed: {error}") from error
    if result.returncode != 0 or not output.is_file():
        detail = result.stderr.strip() or "Piper did not produce a WAV."
        raise RuntimeError(f"Piper synthesis failed: {detail}")


def _load_mms_model(model_root: Path, device: torch.device) -> tuple[Any, Any]:
    """Load the vendor API locally; transformer stubs do not express AutoTokenizer's return type."""

    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    model = cast(Any, VitsModel.from_pretrained(
        model_root, local_files_only=True, use_safetensors=True
    )).to(device)
    model.eval()
    return tokenizer, model


def _synthesize_mms(
    tokenizer: Any,
    model: Any,
    text: str,
    seed: int,
    output: Path,
) -> None:
    torch.manual_seed(seed)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    with torch.inference_mode():
        waveform = model(**inputs).waveform.squeeze().detach().cpu().numpy()
    sf.write(output, waveform, model.config.sampling_rate, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an auditable KSC-text Kazakh spoof-only raw manifest locally."
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", default="20260810")
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--piper-binary", default="piper")
    parser.add_argument("--mms-device", default="auto")
    arguments = parser.parse_args()

    try:
        slice_name = _safe_slice_name(arguments.slice_name)
        if arguments.output_manifest.exists():
            raise ValueError(
                f"Refusing to overwrite existing manifest: {arguments.output_manifest}"
            )
        if not arguments.output_manifest.parent.is_dir():
            raise ValueError(f"Manifest parent does not exist: {arguments.output_manifest.parent}")
        data_root = arguments.data_root.resolve(strict=True)
        output_directory = data_root / "raw" / KSC_DERIVED_KK_SOURCE_ID / "slices" / slice_name
        if output_directory.exists():
            raise ValueError(f"Refusing to overwrite existing generated slice: {output_directory}")
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which(arguments.piper_binary) is None:
            raise ValueError(f"Piper binary is not available on PATH: {arguments.piper_binary}")
        base_rows = load_manifest(arguments.base_manifest)
        validate_manifest(base_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        selected = select_ksc_bonafide_rows(base_rows, limit=arguments.limit, seed=arguments.seed)
        require_valid_assets(selected, data_root)
        lock = load_research_tts_model_lock(arguments.model_lock)
        verified_models = verify_research_tts_model_lock(arguments.model_root, lock)
        profiles = synthesis_profiles(lock)
        if KSC_DERIVED_KK_SOURCE_ID not in ledger:
            raise LicenseLedgerError(
                [f"Missing source {KSC_DERIVED_KK_SOURCE_ID!r} in the license ledger."]
            )
        mms_device = _mms_device(arguments.mms_device)
        model_roots = {
            model.model_id: arguments.model_root.resolve(strict=True) / model.destination
            for model in lock.models
        }
        mms_runtime: tuple[Any, Any] | None = None
        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-ksc-derived-", dir=output_directory.parent)
        )
        staged_manifest_directory = Path(
            tempfile.mkdtemp(
                prefix="kds-ksc-derived-manifest-", dir=arguments.output_manifest.parent
            )
        )
        try:
            rows: list[ManifestRow] = []
            for base_row, profile in assign_synthesis_profiles(selected, profiles):
                transcript = load_verified_ksc_transcript(base_row, arguments.transcript_root)
                relative_asset_path = (
                    Path("raw")
                    / KSC_DERIVED_KK_SOURCE_ID
                    / "slices"
                    / slice_name
                    / profile.model.model_id
                    / f"{base_row.sample_id.rsplit(':', maxsplit=1)[-1]}.wav"
                )
                staged_asset = stage_directory / relative_asset_path.relative_to(
                    Path("raw") / KSC_DERIVED_KK_SOURCE_ID / "slices" / slice_name
                )
                staged_asset.parent.mkdir(parents=True, exist_ok=True)
                seed = synthesis_seed(arguments.seed, base_row, profile)
                paths = verified_models[profile.model.model_id]
                if profile.speaker_id is not None:
                    _synthesize_piper(
                        arguments.piper_binary,
                        profile.model,
                        paths,
                        profile.speaker_id,
                        transcript,
                        staged_asset,
                    )
                    device = "local_cpu_piper"
                else:
                    if mms_runtime is None:
                        mms_runtime = _load_mms_model(
                            model_roots[profile.model.model_id], mms_device
                        )
                    _synthesize_mms(*mms_runtime, transcript, seed, staged_asset)
                    device = f"local_{mms_device.type}_mms"
                info = sf.info(str(staged_asset))
                if info.duration <= 0 or info.samplerate <= 0 or str(info.format).lower() != "wav":
                    raise RuntimeError(f"Generator produced invalid WAV: {staged_asset}")
                rows.append(
                    derived_spoof_row(
                        base_row=base_row,
                        profile=profile,
                        relative_path=relative_asset_path.as_posix(),
                        sha256=sha256_file(staged_asset),
                        duration_s=float(info.duration),
                        original_sr=int(info.samplerate),
                        created_at=arguments.created_at,
                        device=device,
                        seed=seed,
                    )
                )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            staged_manifest = staged_manifest_directory / arguments.output_manifest.name
            write_manifest(staged_manifest, rows)
            if output_directory.exists() or arguments.output_manifest.exists():
                raise ValueError(
                    "Output appeared while synthesis was staging; refusing publication."
                )
            stage_directory.replace(output_directory)
            staged_manifest.replace(arguments.output_manifest)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
            shutil.rmtree(staged_manifest_directory, ignore_errors=True)
    except (
        LicenseLedgerError,
        ManifestError,
        ResearchTtsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (LicenseLedgerError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(rows),
                "label_counts": dict(Counter(row.label for row in rows)),
                "generator_families": dict(Counter(row.generator_family for row in rows)),
                "voices": dict(Counter(row.voice_id for row in rows)),
                "manifest": str(arguments.output_manifest),
                "assets": str(output_directory),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
