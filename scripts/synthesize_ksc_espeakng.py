from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.espeakng import (
    ESPEAKNG_SOURCE_ID,
    ESPEAKNG_SOURCE_LICENSE,
    assign_espeakng_profiles,
    extract_verified_espeakng_runtime,
    load_espeakng_runtime,
    synthesize_espeakng,
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
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)


def _safe_slice_name(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a pinned non-cloning eSpeak NG Kazakh spoof-only raw manifest locally."
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
    parser.add_argument("--seed", default="20260817")
    parser.add_argument("--created-at", required=True)
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
        output_directory = data_root / "raw" / ESPEAKNG_SOURCE_ID / "slices" / slice_name
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
            raise ResearchTtsError("eSpeak NG lock must contain exactly one model.")
        model = lock.models[0]
        runtime = load_espeakng_runtime(model)
        verified_models = verify_research_tts_model_lock(arguments.model_root, lock)
        if ESPEAKNG_SOURCE_ID not in ledger:
            raise LicenseLedgerError(
                [f"Missing source {ESPEAKNG_SOURCE_ID!r} in the license ledger."]
            )
        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-espeakng-", dir=output_directory.parent)
        )
        staged_manifest_directory = Path(
            tempfile.mkdtemp(prefix="kds-espeakng-manifest-", dir=arguments.output_manifest.parent)
        )
        try:
            with tempfile.TemporaryDirectory(prefix="kds-espeakng-runtime-") as runtime_directory:
                runtime_paths = extract_verified_espeakng_runtime(
                    verified_models[model.model_id], runtime, Path(runtime_directory) / "runtime"
                )
                rows: list[ManifestRow] = []
                for base_row, profile in assign_espeakng_profiles(selected, runtime):
                    synthesis_profile = SynthesisProfile(
                        model=model, voice_id=profile.voice_id, speaker_id=None
                    )
                    transcript = load_verified_ksc_transcript(base_row, arguments.transcript_root)
                    relative_asset_path = (
                        Path("raw")
                        / ESPEAKNG_SOURCE_ID
                        / "slices"
                        / slice_name
                        / model.model_id
                        / f"{base_row.sample_id.rsplit(':', maxsplit=1)[-1]}.wav"
                    )
                    staged_asset = stage_directory / relative_asset_path.relative_to(
                        Path("raw") / ESPEAKNG_SOURCE_ID / "slices" / slice_name
                    )
                    staged_asset.parent.mkdir(parents=True, exist_ok=True)
                    synthesize_espeakng(
                        runtime_paths=runtime_paths,
                        runtime=runtime,
                        profile=profile,
                        text=transcript,
                        output=staged_asset,
                    )
                    info = sf.info(str(staged_asset))
                    if (
                        info.duration <= 0
                        or info.samplerate != runtime.sample_rate
                        or str(info.format).lower() != "wav"
                    ):
                        raise RuntimeError(f"eSpeak NG produced invalid WAV: {staged_asset}")
                    seed = synthesis_seed(arguments.seed, base_row, synthesis_profile)
                    rows.append(
                        derived_spoof_row(
                            base_row=base_row,
                            profile=synthesis_profile,
                            relative_path=relative_asset_path.as_posix(),
                            sha256=sha256_file(staged_asset),
                            duration_s=float(info.duration),
                            original_sr=int(info.samplerate),
                            created_at=arguments.created_at,
                            device="local_cpu_espeakng_formant",
                            seed=seed,
                            source_name=ESPEAKNG_SOURCE_ID,
                            source_license=ESPEAKNG_SOURCE_LICENSE,
                            include_tts_seed=False,
                            capture_route="offline_formant_tts",
                        )
                    )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            staged_manifest = staged_manifest_directory / arguments.output_manifest.name
            write_manifest(staged_manifest, rows)
            if output_directory.exists() or arguments.output_manifest.exists():
                raise ValueError("Output appeared while eSpeak NG synthesis was staging.")
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
