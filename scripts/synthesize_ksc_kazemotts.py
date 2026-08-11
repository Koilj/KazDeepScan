from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.kazemotts import (
    KAZEMOTTS_SOURCE_ID,
    KAZEMOTTS_SOURCE_LICENSE,
    KazEmoTtsRuntime,
    assign_kazemotts_profiles,
    extract_verified_kazemotts_source,
    extract_verified_zip_member,
    load_kazemotts_runtime,
)
from kds.data.ksc_derived_kk import (
    SynthesisProfile,
    derived_spoof_row,
    load_verified_ksc_transcript,
    select_ksc_bonafide_rows,
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


class _AttributeMapping(dict[str, object]):
    """Minimal local equivalent of the upstream attrdict dependency."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _safe_slice_name(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("KazEmoTTS CUDA was requested but is unavailable.")
    return device


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ResearchTtsError(f"KazEmoTTS {label} must be a mapping.")
    return cast(Mapping[str, object], value)


def _state_dict(value: object, key: str, label: str) -> Mapping[str, Any]:
    checkpoint = _mapping(value, label)
    state = checkpoint.get(key)
    if not isinstance(state, Mapping):
        raise ResearchTtsError(f"KazEmoTTS {label} is missing tensor mapping {key!r}.")
    return cast(Mapping[str, Any], state)


def _load_kazemotts_models(
    *,
    model: ResearchTtsModel,
    runtime: KazEmoTtsRuntime,
    verified_paths: Mapping[str, Path],
    workspace: Path,
    device: torch.device,
) -> tuple[Any, Any, Callable[[str], tuple[Any, Any]]]:
    """Load only pinned source/checkpoint bytes; legacy pickle loading is explicitly disabled."""

    source_root = extract_verified_kazemotts_source(
        verified_paths[runtime.source_archive_path], runtime, workspace / "source"
    )
    tts_checkpoint = workspace / "EMA_grad_10000.pt"
    vocoder_checkpoint = workspace / "g_01720000"
    extract_verified_zip_member(
        verified_paths[runtime.tts_archive_path],
        runtime.tts_checkpoint_member,
        runtime.tts_checkpoint_size_bytes,
        runtime.tts_checkpoint_sha256,
        tts_checkpoint,
    )
    extract_verified_zip_member(
        verified_paths[runtime.vocoder_archive_path],
        runtime.vocoder_checkpoint_member,
        runtime.vocoder_checkpoint_size_bytes,
        runtime.vocoder_checkpoint_sha256,
        vocoder_checkpoint,
    )
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    grad_tts_class = importlib.import_module("model").GradTTSWithEmo
    generator_class = importlib.import_module("models").Generator
    convert_text = importlib.import_module("text").convert_text
    if not callable(convert_text):
        raise ResearchTtsError("KazEmoTTS source convert_text is not callable.")
    try:
        train_config = json.loads(
            (source_root / "configs" / "train_grad.json").read_text(encoding="utf-8")
        )
        hifigan_config = json.loads(
            (source_root / "configs" / "hifigan-config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchTtsError(f"Cannot load pinned KazEmoTTS configs: {error}") from error
    train_mapping = _mapping(train_config, "train config")
    data_mapping = _mapping(train_mapping.get("data"), "train config data")
    if data_mapping.get("sampling_rate") != runtime.sample_rate:
        raise ResearchTtsError("KazEmoTTS locked sample rate does not match its source config.")
    model_config = _mapping(train_mapping.get("model"), "train config model")
    tts = grad_tts_class(**model_config).to(device).eval()
    tts.load_state_dict(
        _state_dict(
            torch.load(tts_checkpoint, map_location="cpu", weights_only=True),
            "model",
            "Grad-TTS checkpoint",
        ),
        strict=True,
    )
    vocoder = generator_class(_AttributeMapping(_mapping(hifigan_config, "HiFi-GAN config")))
    vocoder = vocoder.to(device).eval()
    vocoder.load_state_dict(
        _state_dict(
            torch.load(vocoder_checkpoint, map_location="cpu", weights_only=True),
            "generator",
            "HiFi-GAN checkpoint",
        ),
        strict=True,
    )
    vocoder.remove_weight_norm()
    return tts, vocoder, cast(Callable[[str], tuple[Any, Any]], convert_text)


def _synthesize(
    *,
    tts: Any,
    vocoder: Any,
    convert_text: Callable[[str], tuple[Any, Any]],
    text: str,
    speaker_id: int,
    emotion_id: int,
    seed: int,
    runtime: KazEmoTtsRuntime,
    device: torch.device,
    output: Path,
) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    text_padded, text_length = convert_text(text)
    if int(text_length.item()) <= 0:
        raise RuntimeError("KazEmoTTS text normalizer produced no supported tokens.")
    with torch.inference_mode():
        _encoded, mel, _attention = tts(
            text_padded.to(device),
            text_length.to(device),
            n_timesteps=runtime.n_timesteps,
            temperature=runtime.temperature,
            stoc=False,
            spk=torch.tensor([speaker_id], device=device),
            emo=torch.tensor([emotion_id], device=device),
            classifier_free_guidance=runtime.classifier_free_guidance,
        )
        waveform = vocoder(mel).squeeze().detach().cpu().numpy()
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("KazEmoTTS produced an empty or non-finite waveform.")
    sf.write(output, waveform, runtime.sample_rate, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a pinned KazEmoTTS Kazakh spoof-only raw manifest locally."
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--limit", type=int, default=450)
    parser.add_argument("--seed", default="20260810")
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    try:
        slice_name = _safe_slice_name(arguments.slice_name)
        if arguments.output_manifest.exists():
            raise ValueError(
                f"Refusing to overwrite existing manifest: {arguments.output_manifest}"
            )
        if not arguments.output_manifest.parent.is_dir():
            raise ValueError(f"Manifest parent does not exist: {arguments.output_manifest.parent}")
        if arguments.limit <= 0:
            raise ValueError("limit must be positive.")
        data_root = arguments.data_root.resolve(strict=True)
        output_directory = data_root / "raw" / KAZEMOTTS_SOURCE_ID / "slices" / slice_name
        if output_directory.exists():
            raise ValueError(f"Refusing to overwrite generated slice: {output_directory}")
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        base_rows = load_manifest(arguments.base_manifest)
        validate_manifest(base_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        selected = select_ksc_bonafide_rows(base_rows, limit=arguments.limit, seed=arguments.seed)
        require_valid_assets(selected, data_root)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("KazEmoTTS lock must contain exactly one model.")
        model = lock.models[0]
        runtime = load_kazemotts_runtime(model)
        verified_models = verify_research_tts_model_lock(arguments.model_root, lock)
        if KAZEMOTTS_SOURCE_ID not in ledger:
            raise LicenseLedgerError(
                [f"Missing source {KAZEMOTTS_SOURCE_ID!r} in the license ledger."]
            )
        device = _device(arguments.device)
        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-kazemotts-", dir=output_directory.parent)
        )
        staged_manifest_directory = Path(
            tempfile.mkdtemp(prefix="kds-kazemotts-manifest-", dir=arguments.output_manifest.parent)
        )
        try:
            with tempfile.TemporaryDirectory(prefix="kds-kazemotts-runtime-") as runtime_directory:
                tts, vocoder, convert_text = _load_kazemotts_models(
                    model=model,
                    runtime=runtime,
                    verified_paths=verified_models[model.model_id],
                    workspace=Path(runtime_directory),
                    device=device,
                )
                rows: list[ManifestRow] = []
                for base_row, profile in assign_kazemotts_profiles(selected, runtime):
                    synthesis_profile = SynthesisProfile(
                        model=model, voice_id=profile.voice_id, speaker_id=profile.speaker_id
                    )
                    transcript = load_verified_ksc_transcript(base_row, arguments.transcript_root)
                    relative_asset_path = (
                        Path("raw")
                        / KAZEMOTTS_SOURCE_ID
                        / "slices"
                        / slice_name
                        / model.model_id
                        / f"{base_row.sample_id.rsplit(':', maxsplit=1)[-1]}.wav"
                    )
                    staged_asset = stage_directory / relative_asset_path.relative_to(
                        Path("raw") / KAZEMOTTS_SOURCE_ID / "slices" / slice_name
                    )
                    staged_asset.parent.mkdir(parents=True, exist_ok=True)
                    seed = synthesis_seed(arguments.seed, base_row, synthesis_profile)
                    _synthesize(
                        tts=tts,
                        vocoder=vocoder,
                        convert_text=convert_text,
                        text=transcript,
                        speaker_id=profile.speaker_id,
                        emotion_id=profile.emotion_id,
                        seed=seed,
                        runtime=runtime,
                        device=device,
                        output=staged_asset,
                    )
                    info = sf.info(str(staged_asset))
                    if (
                        info.duration <= 0
                        or info.samplerate != runtime.sample_rate
                        or str(info.format).lower() != "wav"
                    ):
                        raise RuntimeError(f"KazEmoTTS produced invalid WAV: {staged_asset}")
                    rows.append(
                        derived_spoof_row(
                            base_row=base_row,
                            profile=synthesis_profile,
                            relative_path=relative_asset_path.as_posix(),
                            sha256=sha256_file(staged_asset),
                            duration_s=float(info.duration),
                            original_sr=int(info.samplerate),
                            created_at=arguments.created_at,
                            device=f"local_{device.type}_kazemotts_gradtts_hifigan",
                            seed=seed,
                            source_name=KAZEMOTTS_SOURCE_ID,
                            source_license=KAZEMOTTS_SOURCE_LICENSE,
                            include_tts_seed=True,
                        )
                    )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            staged_manifest = staged_manifest_directory / arguments.output_manifest.name
            write_manifest(staged_manifest, rows)
            if output_directory.exists() or arguments.output_manifest.exists():
                raise ValueError("Output appeared while KazEmoTTS synthesis was staging.")
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
