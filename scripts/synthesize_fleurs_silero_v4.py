"""Create a text-only, fixed-profile Silero V4 spoof candidate from ready FLEURS rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.fleurs import FleursIngestionError, FleursRecord, inspect_fleurs_release
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
from kds.data.silero_v4 import (
    SILERO_V4_SOURCE_ID,
    SILERO_V4_SOURCE_LICENSE,
    SileroV4Error,
    assign_silero_v4_profiles,
    load_silero_v4_model,
    load_silero_v4_runtime,
    normalize_silero_v4_text,
    silero_v4_spoof_row,
    synthesize_silero_v4,
)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Silero V4 CUDA was requested but is unavailable.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Silero V4 device must be CPU or CUDA.")
    return device


def _safe_slice_name(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("slice-name may contain only letters, numbers, hyphens, and underscores.")
    return value


def _load_base_rows(manifests: Iterable[Path]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    seen_sample_ids: set[str] = set()
    expected_sources = {"google_fleurs_ru_v1": "ru", "google_fleurs_kk_v1": "kk"}
    for manifest_path in manifests:
        manifest_rows = load_manifest(manifest_path)
        validate_manifest(manifest_rows)
        for row in manifest_rows:
            expected_language = expected_sources.get(row.source_name)
            if (
                expected_language is None
                or row.language != expected_language
                or row.split != "test"
                or row.label != "bonafide"
                or row.code_switch != "false"
                or row.codec != "wav"
                or not row.relative_path.startswith("processed/")
            ):
                raise ValueError(
                    "Every base row must be a ready, non-code-switched FLEURS ru/kk "
                    "test bona-fide WAV."
                )
            if row.sample_id in seen_sample_ids:
                raise ValueError(
                    f"Duplicate base sample_id across FLEURS manifests: {row.sample_id!r}"
                )
            seen_sample_ids.add(row.sample_id)
            rows.append(row)
    if {row.language for row in rows} != {"ru", "kk"}:
        raise ValueError("Exactly one or more ready FLEURS rows are required for both ru and kk.")
    return rows


def _verified_transcripts(release_root: Path, rows: Iterable[ManifestRow]) -> dict[str, str]:
    rows = list(rows)
    by_language: dict[str, list[ManifestRow]] = {"ru": [], "kk": []}
    for row in rows:
        by_language[row.language].append(row)
    result: dict[str, str] = {}
    locales = {"ru": "ru_ru", "kk": "kk_kz"}
    for language, locale in locales.items():
        if not by_language[language]:
            continue
        _report, records_by_split = inspect_fleurs_release(release_root, locale)
        records = {
            record.filename.removesuffix(".wav"): record for record in records_by_split["test"]
        }
        for row in by_language[language]:
            source_prefix = f"{row.source_name}:"
            if not row.sample_id.startswith(source_prefix):
                raise ValueError(f"FLEURS sample has an invalid source prefix: {row.sample_id!r}")
            filename_stem = row.sample_id.removeprefix(source_prefix)
            record = records.get(filename_stem)
            if record is None:
                raise ValueError(f"FLEURS test transcript is missing for {row.sample_id!r}")
            _verify_record_against_row(record, row)
            result[row.sample_id] = record.transcript
    if len(result) != len(rows):
        raise ValueError("FLEURS transcript verification did not cover every base row.")
    return result


def _verify_record_against_row(record: FleursRecord, row: ManifestRow) -> None:
    expected_text_id = f"{row.source_name}:prompt:{record.prompt_id}"
    if row.text_id != expected_text_id or row.text_hash != record.text_hash:
        raise ValueError(
            f"FLEURS transcript provenance mismatch for {row.sample_id!r}; refusing synthesis."
        )


def _text_rejection(row: ManifestRow, text: str) -> dict[str, str]:
    try:
        normalize_silero_v4_text(text)
    except SileroV4Error as error:
        return {
            "sample_id": row.sample_id,
            "text_id": row.text_id,
            "text_hash": row.text_hash,
            "reason": str(error),
        }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate only fixed-profile, text-compatible FLEURS RU/KK Silero V4 raw spoof "
            "assets. This command has no reference-audio, voice-path or random-profile option."
        )
    )
    parser.add_argument("--base-manifest", type=Path, action="append", required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--text-rejection-report", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    try:
        slice_name = _safe_slice_name(arguments.slice_name)
        if arguments.output_manifest.exists() or arguments.text_rejection_report.exists():
            raise ValueError("Refusing to overwrite Silero V4 manifest or text-rejection report.")
        if (
            not arguments.output_manifest.parent.is_dir()
            or not arguments.text_rejection_report.parent.is_dir()
            or not arguments.data_root.is_dir()
        ):
            raise ValueError("Output parents and data-root must already exist.")
        data_root = arguments.data_root.resolve(strict=True)
        output_directory = data_root / "raw" / SILERO_V4_SOURCE_ID / "slices" / slice_name
        if output_directory.exists():
            raise ValueError(f"Refusing to overwrite generated slice: {output_directory}")
        output_directory.parent.mkdir(parents=True, exist_ok=True)

        base_rows = _load_base_rows(arguments.base_manifest)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        if SILERO_V4_SOURCE_ID not in ledger:
            raise LicenseLedgerError(
                [f"Missing source {SILERO_V4_SOURCE_ID!r} in the license ledger."]
            )
        require_valid_assets(base_rows, data_root)
        transcripts = _verified_transcripts(arguments.release_root, base_rows)

        text_rejections = [
            rejection
            for row in base_rows
            if (rejection := _text_rejection(row, transcripts[row.sample_id]))
        ]
        rejected_sample_ids = {rejection["sample_id"] for rejection in text_rejections}
        eligible_rows = [row for row in base_rows if row.sample_id not in rejected_sample_ids]
        if not eligible_rows:
            raise ValueError("No FLEURS rows are compatible with the Silero V4 text contract.")

        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Silero V4 lock must contain exactly one model.")
        model_spec = lock.models[0]
        runtime = load_silero_v4_runtime(model_spec)
        verified_models = verify_research_tts_model_lock(arguments.model_root, lock)
        verified_paths = verified_models[model_spec.model_id]
        if runtime.source_archive_path not in verified_paths:
            raise ResearchTtsError("Silero V4 source archive is missing from the verified lock.")
        assignments = assign_silero_v4_profiles(eligible_rows, runtime)
        device = _device(arguments.device)

        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-silero-v4-", dir=output_directory.parent)
        )
        staged_manifest_directory = Path(
            tempfile.mkdtemp(prefix="kds-silero-v4-manifest-", dir=arguments.output_manifest.parent)
        )
        staged_report_directory = Path(
            tempfile.mkdtemp(
                prefix="kds-silero-v4-report-", dir=arguments.text_rejection_report.parent
            )
        )
        try:
            model = load_silero_v4_model(verified_paths[runtime.package_path], runtime, device)
            rows: list[ManifestRow] = []
            for base_row, profile in assignments:
                source_suffix = hashlib.sha256(base_row.sample_id.encode()).hexdigest()[:16]
                relative_asset_path = (
                    Path("raw")
                    / SILERO_V4_SOURCE_ID
                    / "slices"
                    / slice_name
                    / model_spec.model_id
                    / base_row.language
                    / f"{source_suffix}.wav"
                )
                staged_asset = stage_directory / relative_asset_path.relative_to(
                    Path("raw") / SILERO_V4_SOURCE_ID / "slices" / slice_name
                )
                staged_asset.parent.mkdir(parents=True, exist_ok=True)
                synthesize_silero_v4(
                    model=model,
                    profile=profile,
                    text=transcripts[base_row.sample_id],
                    runtime=runtime,
                    output=staged_asset,
                )
                info = sf.info(str(staged_asset))
                if (
                    info.duration <= 0
                    or info.samplerate != runtime.sample_rate
                    or str(info.format).lower() != "wav"
                ):
                    raise RuntimeError(f"Silero V4 produced an invalid WAV: {staged_asset}")
                rows.append(
                    silero_v4_spoof_row(
                        base_row=base_row,
                        model=model_spec,
                        profile=profile,
                        relative_path=relative_asset_path.as_posix(),
                        sha256=sha256_file(staged_asset),
                        duration_s=float(info.duration),
                        original_sr=int(info.samplerate),
                        created_at=arguments.created_at,
                        device=f"local_{device.type}_silero_v4_fastpitch_hifigan",
                    )
                )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            staged_manifest = staged_manifest_directory / arguments.output_manifest.name
            write_manifest(staged_manifest, rows)
            report_payload = {
                "base_manifests": [str(path) for path in arguments.base_manifest],
                "base_manifest_sha256": {
                    str(path): sha256_file(path) for path in arguments.base_manifest
                },
                "model_lock": str(arguments.model_lock),
                "model_lock_sha256": sha256_file(arguments.model_lock),
                "published_rows": len(rows),
                "rejected_rows": text_rejections,
            }
            staged_report = staged_report_directory / arguments.text_rejection_report.name
            with staged_report.open("x", encoding="utf-8") as handle:
                json.dump(report_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            if (
                output_directory.exists()
                or arguments.output_manifest.exists()
                or arguments.text_rejection_report.exists()
            ):
                raise ValueError("A Silero V4 output appeared while publication was staging.")
            stage_directory.replace(output_directory)
            staged_manifest.replace(arguments.output_manifest)
            staged_report.replace(arguments.text_rejection_report)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
            shutil.rmtree(staged_manifest_directory, ignore_errors=True)
            shutil.rmtree(staged_report_directory, ignore_errors=True)
    except (
        FleursIngestionError,
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
            if isinstance(error, (LicenseLedgerError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "published": len(rows),
                "text_rejected": len(text_rejections),
                "source_license": SILERO_V4_SOURCE_LICENSE,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
