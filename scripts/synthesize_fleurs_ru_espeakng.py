"""Generate a pinned, text-only Russian eSpeak NG raw spoof layer from FLEURS RU."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.espeakng import (
    EspeakNgRuntime,
    extract_verified_espeakng_runtime,
    load_espeakng_runtime,
    synthesize_espeakng,
)
from kds.data.fleurs import FleursIngestionError, verified_fleurs_test_transcripts
from kds.data.fleurs_espeakng import (
    FLEURS_RU_ESPEAKNG_SOURCE_ID,
    FLEURS_RU_SOURCE_ID,
    FleursEspeakNgError,
    fleurs_ru_espeakng_spoof_row,
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


def _base_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    # Kept small so runtime selection cannot silently accept another source/language.
    if not rows or any(
        row.source_name != FLEURS_RU_SOURCE_ID
        or row.language != "ru"
        or row.split != "test"
        or row.label != "bonafide"
        or row.code_switch != "false"
        or row.codec != "wav"
        or not row.relative_path.startswith("processed/")
        for row in rows
    ):
        raise FleursEspeakNgError(
            "Base manifest must contain only ready FLEURS RU test bona-fide WAV rows."
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()

    try:
        slice_name = _safe_slice_name(arguments.slice_name)
        if arguments.output_manifest.exists():
            raise ValueError(
                f"Refusing to overwrite existing manifest: {arguments.output_manifest}"
            )
        if not arguments.output_manifest.parent.is_dir() or not arguments.data_root.is_dir():
            raise ValueError("Manifest output parent and data-root must already exist.")
        data_root = arguments.data_root.resolve(strict=True)
        output_directory = data_root / "raw" / FLEURS_RU_ESPEAKNG_SOURCE_ID / "slices" / slice_name
        if output_directory.exists():
            raise ValueError(f"Refusing to overwrite generated slice: {output_directory}")
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        base = load_manifest(arguments.base_manifest)
        validate_manifest(base)
        base_rows = _base_rows(list(base))
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base, ledger)
        if FLEURS_RU_ESPEAKNG_SOURCE_ID not in ledger:
            raise LicenseLedgerError(
                [f"Missing source {FLEURS_RU_ESPEAKNG_SOURCE_ID!r} in ledger."]
            )
        require_valid_assets(base, data_root)
        transcripts = verified_fleurs_test_transcripts(arguments.release_root, base)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Russian eSpeak NG lock must contain exactly one model.")
        model = lock.models[0]
        runtime: EspeakNgRuntime = load_espeakng_runtime(model)
        if runtime.voice != "ru":
            raise ResearchTtsError("Russian eSpeak NG synthesis requires pinned voice='ru'.")
        verified = verify_research_tts_model_lock(arguments.model_root, lock)[model.model_id]
        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-fleurs-ru-espeakng-", dir=output_directory.parent)
        )
        staged_manifest_directory = Path(
            tempfile.mkdtemp(
                prefix="kds-fleurs-ru-espeakng-manifest-", dir=arguments.output_manifest.parent
            )
        )
        try:
            with tempfile.TemporaryDirectory(prefix="kds-fleurs-ru-espeakng-runtime-") as temp_root:
                runtime_paths = extract_verified_espeakng_runtime(
                    verified, runtime, Path(temp_root) / "runtime"
                )
                rows: list[ManifestRow] = []
                for index, base_row in enumerate(base_rows):
                    profile = runtime.profiles[index % len(runtime.profiles)]
                    suffix = hashlib.sha256(base_row.sample_id.encode()).hexdigest()[:16]
                    relative_path = (
                        Path("raw")
                        / FLEURS_RU_ESPEAKNG_SOURCE_ID
                        / "slices"
                        / slice_name
                        / model.model_id
                        / f"{suffix}.wav"
                    )
                    staged_asset = stage_directory / relative_path.relative_to(
                        Path("raw") / FLEURS_RU_ESPEAKNG_SOURCE_ID / "slices" / slice_name
                    )
                    staged_asset.parent.mkdir(parents=True, exist_ok=True)
                    synthesize_espeakng(
                        runtime_paths=runtime_paths,
                        runtime=runtime,
                        profile=profile,
                        text=transcripts[base_row.sample_id],
                        output=staged_asset,
                    )
                    info = sf.info(str(staged_asset))
                    if (
                        info.duration <= 0
                        or info.samplerate != runtime.sample_rate
                        or str(info.format).lower() != "wav"
                    ):
                        raise RuntimeError(f"eSpeak NG produced invalid WAV: {staged_asset}")
                    rows.append(
                        fleurs_ru_espeakng_spoof_row(
                            base_row=base_row,
                            model=model,
                            profile=profile,
                            relative_path=relative_path.as_posix(),
                            sha256=sha256_file(staged_asset),
                            duration_s=float(info.duration),
                            original_sr=int(info.samplerate),
                            created_at=arguments.created_at,
                        )
                    )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            staged_manifest = staged_manifest_directory / arguments.output_manifest.name
            write_manifest(staged_manifest, rows)
            if output_directory.exists() or arguments.output_manifest.exists():
                raise ValueError("A Russian eSpeak NG output appeared while staging.")
            stage_directory.replace(output_directory)
            staged_manifest.replace(arguments.output_manifest)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
            shutil.rmtree(staged_manifest_directory, ignore_errors=True)
    except (
        FleursEspeakNgError,
        FleursIngestionError,
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
                "voices": dict(Counter(row.voice_id for row in rows)),
                "output_manifest": str(arguments.output_manifest),
                "assets": str(output_directory),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
