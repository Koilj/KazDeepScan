from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kds.data.assets import require_valid_assets, sha256_file
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
from kds.data.sparktts import (
    SPARKTTS_SOURCE_ID,
    SPARKTTS_SOURCE_LICENSE,
    SparkTtsProfile,
    SparkTtsRuntime,
    assign_sparktts_profiles,
    extract_verified_sparktts_source,
    load_sparktts_runtime,
)

_SEMANTIC_TOKEN_PATTERN = re.compile(r"bicodec_semantic_(\d+)")
_GLOBAL_TOKEN_PATTERN = re.compile(r"bicodec_global_(\d+)")
_GLOBAL_TOKEN_COUNT = 32
_GLOBAL_TOKEN_LIMIT = 4**6
_SEMANTIC_TOKEN_LIMIT = 8192
_RETRYABLE_OUTPUT_ERRORS = (
    "Spark-TTS generated no ",
    "Spark-TTS generated an out-of-range ",
    "Spark-TTS controlled output has an unexpected number of global voice tokens.",
)


def _safe_slice_name(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Spark-TTS CUDA was requested but is unavailable.")
    return device


def _bundle_directory(
    verified_paths: Mapping[str, Path], paths: tuple[str, ...], label: str
) -> Path:
    directories = {verified_paths[path].parent for path in paths}
    if len(directories) != 1:
        raise ResearchTtsError(f"Spark-TTS {label} artifacts are not in one directory.")
    return directories.pop()


def _load_sparktts_models(
    *,
    model: ResearchTtsModel,
    runtime: SparkTtsRuntime,
    verified_paths: Mapping[str, Path],
    workspace: Path,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    """Load pinned safe tensors for controlled generation without initializing wav2vec2."""

    source_root = extract_verified_sparktts_source(
        verified_paths[runtime.source_archive_path], runtime, workspace / "source"
    )
    root_config = verified_paths[runtime.root_config_path]
    bicodec_directory = _bundle_directory(
        verified_paths,
        (runtime.bicodec_config_path, runtime.bicodec_checkpoint_path),
        "BiCodec",
    )
    llm_directory = _bundle_directory(
        verified_paths,
        (
            runtime.llm_config_path,
            runtime.llm_merges_path,
            runtime.llm_tokenizer_config_path,
            runtime.llm_vocab_path,
            runtime.llm_checkpoint_path,
        ),
        "LLM",
    )
    if bicodec_directory.name != "BiCodec" or llm_directory.name != "LLM":
        raise ResearchTtsError("Spark-TTS locked artifact directories do not match the runtime.")
    try:
        from omegaconf import OmegaConf

        root_config_value = cast(
            Mapping[str, Any], OmegaConf.to_container(OmegaConf.load(root_config), resolve=True)
        )
        if int(root_config_value["sample_rate"]) != runtime.sample_rate:
            raise ResearchTtsError("Spark-TTS locked sample rate does not match its source config.")
    except (KeyError, OSError, TypeError, ValueError) as error:
        if isinstance(error, ResearchTtsError):
            raise
        raise ResearchTtsError(f"Cannot read Spark-TTS locked root config: {error}") from error

    conflicting_modules = [
        name for name in sys.modules if name == "sparktts" or name.startswith("sparktts.")
    ]
    if conflicting_modules:
        raise ResearchTtsError("Spark-TTS source is already imported; refusing ambiguous runtime.")
    sys.path.insert(0, str(source_root))
    try:
        importlib.invalidate_caches()
        bicodec_class = importlib.import_module("sparktts.models.bicodec").BiCodec
        codec = bicodec_class.load_from_checkpoint(bicodec_directory).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            llm_directory,
            local_files_only=True,
            trust_remote_code=False,
        )
        language_model: Any = AutoModelForCausalLM.from_pretrained(
            llm_directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        language_model = language_model.to(device).eval()
    except (OSError, RuntimeError, ValueError) as error:
        raise ResearchTtsError(
            f"Cannot load pinned Spark-TTS controlled runtime: {error}"
        ) from error
    finally:
        sys.path.remove(str(source_root))
    return cast(Any, tokenizer), cast(Any, language_model), cast(Any, codec)


def _controlled_prompt(profile: SparkTtsProfile, text: str) -> str:
    gender_id = {"female": 0, "male": 1}[profile.gender]
    level_id = {
        "very_low": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "very_high": 4,
    }
    return "".join(
        (
            "<|task_controllable_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            f"<|gender_{gender_id}|>",
            f"<|pitch_label_{level_id[profile.pitch]}|>",
            f"<|speed_label_{level_id[profile.speed]}|>",
            "<|end_style_label|>",
        )
    )


def _token_ids(
    pattern: re.Pattern[str], generated_text: str, *, limit: int, label: str
) -> list[int]:
    token_ids = [int(value) for value in pattern.findall(generated_text)]
    if not token_ids:
        raise RuntimeError(f"Spark-TTS generated no {label} tokens.")
    if any(token_id < 0 or token_id >= limit for token_id in token_ids):
        raise RuntimeError(f"Spark-TTS generated an out-of-range {label} token.")
    return token_ids


def _bicodec_global_tensor(global_ids: list[int], device: torch.device) -> torch.Tensor:
    """Recreate the [batch, control-axis, speaker-token] shape used upstream."""

    return torch.tensor([[global_ids]], dtype=torch.long, device=device)


def _attempt_seed(base_seed: int, attempt: int) -> int:
    """Return a reproducible retry seed only for a structurally invalid sampled stream."""

    if attempt < 0:
        raise ValueError("Spark-TTS generation attempt must not be negative.")
    if attempt == 0:
        return base_seed
    digest = hashlib.sha256(f"{base_seed}:sparktts-attempt:{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _synthesize(
    *,
    tokenizer: Any,
    language_model: Any,
    codec: Any,
    text: str,
    profile: SparkTtsProfile,
    seed: int,
    runtime: SparkTtsRuntime,
    device: torch.device,
    output: Path,
) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    prompt = _controlled_prompt(profile, text)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = language_model.generate(
            **model_inputs,
            max_new_tokens=runtime.max_new_tokens,
            do_sample=True,
            top_k=runtime.top_k,
            top_p=runtime.top_p,
            temperature=runtime.temperature,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated, strict=True)
        ]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        semantic_ids = _token_ids(
            _SEMANTIC_TOKEN_PATTERN,
            generated_text,
            limit=_SEMANTIC_TOKEN_LIMIT,
            label="semantic",
        )
        global_ids = _token_ids(
            _GLOBAL_TOKEN_PATTERN,
            generated_text,
            limit=_GLOBAL_TOKEN_LIMIT,
            label="global",
        )
        if len(global_ids) != _GLOBAL_TOKEN_COUNT:
            raise RuntimeError(
                "Spark-TTS controlled output has an unexpected number of global voice tokens."
            )
        waveform = codec.detokenize(
            torch.tensor([semantic_ids], dtype=torch.long, device=device),
            # The upstream BiCodecTokenizer adds the middle control axis before delegating to
            # BiCodec. We bypass only its wav2vec2 initialization, retaining this [B, 1, T]
            # speaker-token shape exactly.
            _bicodec_global_tensor(global_ids, device),
        )
    samples = waveform.detach().float().cpu().numpy().reshape(-1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError("Spark-TTS produced an empty or non-finite waveform.")
    sf.write(output, samples, runtime.sample_rate, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a pinned non-cloning Spark-TTS Kazakh spoof-only raw manifest locally."
        )
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
        output_directory = data_root / "raw" / SPARKTTS_SOURCE_ID / "slices" / slice_name
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
            raise ResearchTtsError("Spark-TTS lock must contain exactly one model.")
        model = lock.models[0]
        runtime = load_sparktts_runtime(model)
        verified_models = verify_research_tts_model_lock(arguments.model_root, lock)
        if SPARKTTS_SOURCE_ID not in ledger:
            raise LicenseLedgerError(
                [f"Missing source {SPARKTTS_SOURCE_ID!r} in the license ledger."]
            )
        device = _device(arguments.device)
        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-sparktts-", dir=output_directory.parent)
        )
        staged_manifest_directory = Path(
            tempfile.mkdtemp(prefix="kds-sparktts-manifest-", dir=arguments.output_manifest.parent)
        )
        try:
            with tempfile.TemporaryDirectory(prefix="kds-sparktts-runtime-") as runtime_directory:
                tokenizer, language_model, codec = _load_sparktts_models(
                    model=model,
                    runtime=runtime,
                    verified_paths=verified_models[model.model_id],
                    workspace=Path(runtime_directory),
                    device=device,
                )
                rows: list[ManifestRow] = []
                for base_row, assigned_profile in assign_sparktts_profiles(selected, runtime):
                    transcript = load_verified_ksc_transcript(base_row, arguments.transcript_root)
                    relative_asset_path = (
                        Path("raw")
                        / SPARKTTS_SOURCE_ID
                        / "slices"
                        / slice_name
                        / model.model_id
                        / f"{base_row.sample_id.rsplit(':', maxsplit=1)[-1]}.wav"
                    )
                    staged_asset = stage_directory / relative_asset_path.relative_to(
                        Path("raw") / SPARKTTS_SOURCE_ID / "slices" / slice_name
                    )
                    staged_asset.parent.mkdir(parents=True, exist_ok=True)
                    profile_candidates = [
                        assigned_profile,
                        *(
                            profile
                            for profile in runtime.profiles
                            if profile.voice_id != assigned_profile.voice_id
                        ),
                    ][: runtime.profile_attempts]
                    profile_errors: list[str] = []
                    for profile in profile_candidates:
                        synthesis_profile = SynthesisProfile(
                            model=model, voice_id=profile.voice_id, speaker_id=None
                        )
                        base_seed = synthesis_seed(arguments.seed, base_row, synthesis_profile)
                        retry_errors: list[str] = []
                        for attempt in range(runtime.generation_attempts):
                            seed = _attempt_seed(base_seed, attempt)
                            try:
                                _synthesize(
                                    tokenizer=tokenizer,
                                    language_model=language_model,
                                    codec=codec,
                                    text=transcript,
                                    profile=profile,
                                    seed=seed,
                                    runtime=runtime,
                                    device=device,
                                    output=staged_asset,
                                )
                                break
                            except RuntimeError as error:
                                if not str(error).startswith(_RETRYABLE_OUTPUT_ERRORS):
                                    raise
                                retry_errors.append(str(error))
                        else:
                            profile_errors.append(
                                f"{profile.voice_id}: " + " | ".join(retry_errors)
                            )
                            continue
                        break
                    else:
                        raise RuntimeError(
                            "Spark-TTS produced no structurally valid output after fixed "
                            f"profile/seed attempts for {base_row.sample_id!r}: "
                            + " || ".join(profile_errors)
                        )
                    info = sf.info(str(staged_asset))
                    if (
                        info.duration <= 0
                        or info.samplerate != runtime.sample_rate
                        or str(info.format).lower() != "wav"
                    ):
                        raise RuntimeError(f"Spark-TTS produced invalid WAV: {staged_asset}")
                    rows.append(
                        derived_spoof_row(
                            base_row=base_row,
                            profile=synthesis_profile,
                            relative_path=relative_asset_path.as_posix(),
                            sha256=sha256_file(staged_asset),
                            duration_s=float(info.duration),
                            original_sr=int(info.samplerate),
                            created_at=arguments.created_at,
                            device=f"local_{device.type}_sparktts_controlled_bicodec",
                            seed=seed,
                            source_name=SPARKTTS_SOURCE_ID,
                            source_license=SPARKTTS_SOURCE_LICENSE,
                            include_tts_seed=True,
                        )
                    )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            staged_manifest = staged_manifest_directory / arguments.output_manifest.name
            write_manifest(staged_manifest, rows)
            if output_directory.exists() or arguments.output_manifest.exists():
                raise ValueError("Output appeared while Spark-TTS synthesis was staging.")
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
                "source_name": SPARKTTS_SOURCE_ID,
                "profiles": len(runtime.profiles),
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
