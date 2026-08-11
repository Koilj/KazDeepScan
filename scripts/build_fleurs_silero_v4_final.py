"""Freeze one paired FLEURS RU/KK research candidate with an explicit spoof source."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.silero_v4 import SILERO_V4_SOURCE_ID


def _language_source(language: str) -> str:
    return {"ru": "google_fleurs_ru_v1", "kk": "google_fleurs_kk_v1"}[language]


def _selected_base_rows(rows: Iterable[ManifestRow], language: str) -> list[ManifestRow]:
    source_name = _language_source(language)
    selected = list(rows)
    if not selected or any(
        row.source_name != source_name
        or row.split != "test"
        or row.label != "bonafide"
        or row.language != language
        or row.code_switch != "false"
        or not row.relative_path.startswith("processed/")
        for row in selected
    ):
        raise ValueError(
            f"Base manifest must contain only ready FLEURS {language} test bona-fide rows."
        )
    return selected


def _selected_spoof_rows(
    rows: Iterable[ManifestRow], language: str, spoof_source: str
) -> list[ManifestRow]:
    selected = [row for row in rows if row.language == language]
    if not selected or any(
        row.source_name != spoof_source
        or row.split != "test"
        or row.label != "spoof"
        or row.code_switch != "false"
        for row in selected
    ):
        raise ValueError(f"Spoof manifest has invalid {language} rows for {spoof_source!r}.")
    return selected


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} JSON report: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON report must be an object.")
    return cast(dict[str, object], value)


def _report_rejected_rows(report: Mapping[str, object], label: str) -> list[dict[str, object]]:
    rows = report.get("rejected_rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} report rejected_rows must be an array of objects.")
    return [cast(dict[str, object], row) for row in rows]


def _rejected_base_ids(
    report: Mapping[str, object], base_manifest: Path, base_rows: Iterable[ManifestRow]
) -> set[str]:
    expected_hash = sha256_file(base_manifest)
    hashes = report.get("base_manifest_sha256")
    if not isinstance(hashes, dict) or hashes.get(str(base_manifest)) != expected_hash:
        raise ValueError("Text-rejection report is not pinned to the supplied base manifest bytes.")
    base_by_id = {row.sample_id: row for row in base_rows}
    rejected_ids: set[str] = set()
    for rejected in _report_rejected_rows(report, "Text-rejection"):
        sample_id = rejected.get("sample_id")
        text_hash = rejected.get("text_hash")
        if not isinstance(sample_id, str) or not isinstance(text_hash, str):
            raise ValueError("Text-rejection report row needs string sample_id and text_hash.")
        base_row = base_by_id.get(sample_id)
        if base_row is None:
            continue
        if base_row.text_hash != text_hash or sample_id in rejected_ids:
            raise ValueError("Text-rejection report has mismatched or duplicate base provenance.")
        rejected_ids.add(sample_id)
    return rejected_ids


def _rejected_raw_ids(report: Mapping[str, object], raw_manifest: Path) -> set[str]:
    if report.get("input_manifest") != str(raw_manifest):
        raise ValueError(
            "Audio-rejection report does not identify the supplied raw spoof manifest."
        )
    result: set[str] = set()
    for rejected in _report_rejected_rows(report, "Audio-rejection"):
        sample_id = rejected.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in result:
            raise ValueError("Audio-rejection report has an invalid or duplicate sample_id.")
        result.add(sample_id)
    return result


def _by_text(rows: Iterable[ManifestRow], label: str) -> dict[str, ManifestRow]:
    result: dict[str, ManifestRow] = {}
    for row in rows:
        if row.text_hash in result:
            raise ValueError(f"{label} has duplicate text_hash={row.text_hash!r}.")
        result[row.text_hash] = row
    return result


def build_final_rows(
    *,
    base_rows: Iterable[ManifestRow],
    raw_spoof_rows: Iterable[ManifestRow],
    ready_spoof_rows: Iterable[ManifestRow],
    text_rejection_report: Mapping[str, object],
    audio_rejection_report: Mapping[str, object],
    base_manifest: Path,
    raw_manifest: Path,
    language: str,
    spoof_source: str = SILERO_V4_SOURCE_ID,
) -> list[ManifestRow]:
    """Prove every non-paired base/raw row has a published, specific rejection reason."""

    base = _selected_base_rows(base_rows, language)
    raw = _selected_spoof_rows(raw_spoof_rows, language, spoof_source)
    ready = _selected_spoof_rows(ready_spoof_rows, language, spoof_source)
    base_by_text = _by_text(base, "FLEURS base")
    raw_by_text = _by_text(raw, "Raw spoof")
    ready_by_text = _by_text(ready, "Ready spoof")
    text_rejected_ids = _rejected_base_ids(text_rejection_report, base_manifest, base)
    expected_raw_texts = {
        row.text_hash for row in base if row.sample_id not in text_rejected_ids
    }
    if set(raw_by_text) != expected_raw_texts:
        raise ValueError("Raw spoof texts do not exactly equal base rows minus text rejections.")
    if any(
        raw_by_text[text_hash].text_id != base_by_text[text_hash].text_id
        for text_hash in raw_by_text
    ):
        raise ValueError("Raw spoof text_id does not match its FLEURS base row.")

    audio_rejected_ids = _rejected_raw_ids(audio_rejection_report, raw_manifest)
    raw_by_id = {row.sample_id: row for row in raw}
    if not audio_rejected_ids.issubset(raw_by_id):
        raise ValueError("Audio-rejection report names a spoof row absent from the raw manifest.")
    expected_ready_texts = {
        row.text_hash for row in raw if row.sample_id not in audio_rejected_ids
    }
    if set(ready_by_text) != expected_ready_texts:
        raise ValueError("Ready spoof texts do not exactly equal raw rows minus audio rejections.")
    if any(
        ready_by_text[text_hash].sample_id != raw_by_text[text_hash].sample_id
        for text_hash in ready_by_text
    ):
        raise ValueError("Ready spoof row identity differs from its raw spoof source row.")

    ordered_text_hashes = sorted(ready_by_text)
    combined = [base_by_text[text_hash] for text_hash in ordered_text_hashes] + [
        ready_by_text[text_hash] for text_hash in ordered_text_hashes
    ]
    if len(combined) != 2 * len(ordered_text_hashes):
        raise ValueError("FLEURS paired research candidate is not binary paired.")
    return combined


def _reject_overlap(rows: Iterable[ManifestRow], excluded_manifests: Iterable[Path]) -> None:
    candidate_rows = list(rows)
    candidate_samples = {row.sample_id for row in candidate_rows}
    candidate_assets = {row.sha256 for row in candidate_rows}
    candidate_texts = {row.text_hash for row in candidate_rows}
    for manifest_path in excluded_manifests:
        excluded = load_manifest(manifest_path)
        validate_manifest(excluded)
        overlap = {
            "sample_id": candidate_samples.intersection(row.sample_id for row in excluded),
            "sha256": candidate_assets.intersection(row.sha256 for row in excluded),
            "text_hash": candidate_texts.intersection(row.text_hash for row in excluded),
        }
        nonempty = {key: values for key, values in overlap.items() if values}
        if nonempty:
            rendered = ", ".join(f"{key}={len(values)}" for key, values in nonempty.items())
            raise ValueError(
                f"Final candidate overlaps excluded manifest {manifest_path}: {rendered}."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish one binary paired FLEURS RU/KK research candidate with accounted text and "
            "audio rejections. It does not run detector inference."
        )
    )
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--spoof-raw-manifest", type=Path, required=True)
    parser.add_argument("--spoof-ready-manifest", type=Path, required=True)
    parser.add_argument("--text-rejection-report", type=Path, required=True)
    parser.add_argument("--audio-rejection-report", type=Path, required=True)
    parser.add_argument("--language", choices=("ru", "kk"), required=True)
    parser.add_argument("--spoof-source", default=SILERO_V4_SOURCE_ID)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        if arguments.output_manifest.exists():
            raise ValueError(
                f"Refusing to overwrite final-candidate manifest: {arguments.output_manifest}"
            )
        if not arguments.output_manifest.parent.is_dir() or not arguments.data_root.is_dir():
            raise ValueError("Final-candidate output parent and data-root must already exist.")
        base_rows = load_manifest(arguments.base_manifest)
        raw_spoof_rows = load_manifest(arguments.spoof_raw_manifest)
        ready_spoof_rows = load_manifest(arguments.spoof_ready_manifest)
        validate_manifest(base_rows)
        validate_manifest(raw_spoof_rows)
        validate_manifest(ready_spoof_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        validate_manifest_licenses(raw_spoof_rows, ledger)
        validate_manifest_licenses(ready_spoof_rows, ledger)
        final_rows = build_final_rows(
            base_rows=base_rows,
            raw_spoof_rows=raw_spoof_rows,
            ready_spoof_rows=ready_spoof_rows,
            text_rejection_report=_json_object(arguments.text_rejection_report, "Text-rejection"),
            audio_rejection_report=_json_object(
                arguments.audio_rejection_report, "Audio-rejection"
            ),
            base_manifest=arguments.base_manifest,
            raw_manifest=arguments.spoof_raw_manifest,
            language=arguments.language,
            spoof_source=arguments.spoof_source,
        )
        validate_manifest(final_rows)
        validate_manifest_licenses(final_rows, ledger)
        _reject_overlap(final_rows, arguments.exclude_manifest)
        require_valid_assets(final_rows, arguments.data_root)
        write_manifest(arguments.output_manifest, final_rows)
    except (LicenseLedgerError, ManifestError, OSError, ValueError) as error:
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
                "language": arguments.language,
                "rows": len(final_rows),
                "pairs": len(final_rows) // 2,
                "manifest": str(arguments.output_manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
