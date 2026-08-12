"""Materialize the frozen Stage-C bona-fide selection and account all RU QA rejections."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.assets import require_valid_assets, sha256_file
from kds.data.fleurs import (
    FleursExtractedAsset,
    FleursIngestionError,
    extract_fleurs_audio_slice,
    fleurs_manifest_rows,
    inspect_extracted_fleurs_audio,
    inspect_fleurs_release,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.preprocess import preprocess_rows
from kds.eval.fresh_suite_selection import FreshSuiteSelectionError, load_fresh_suite_selection


def _role_items(plan: Mapping[str, object], language: str) -> list[Mapping[str, object]]:
    roles = cast(Mapping[str, object], plan["roles"])
    role = cast(Mapping[str, object], roles[language])
    return [cast(Mapping[str, object], item) for item in cast(list[object], role["items"])]


def _load_manifests(paths: list[Path]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for path in paths:
        current = load_manifest(path)
        validate_manifest(current)
        rows.extend(current)
    return rows


def _selected_ready_rows(
    items: list[Mapping[str, object]], pool: list[ManifestRow], language: str
) -> list[ManifestRow]:
    by_id = {row.sample_id: row for row in pool}
    selected: list[ManifestRow] = []
    for item in items:
        sample_id = cast(str, item["sample_id"])
        row = by_id.get(sample_id)
        base = item.get("base_asset")
        if row is None or not isinstance(base, dict):
            raise FreshSuiteSelectionError(
                f"Stage-C {language} selected base row is unavailable: {sample_id!r}."
            )
        expected = {
            "relative_path": row.relative_path,
            "sha256": row.sha256,
            "duration_s": row.duration_s,
            "original_sr": row.original_sr,
            "codec": row.codec,
        }
        if base != expected or row.language != language or row.text_hash != item["text_hash"]:
            raise FreshSuiteSelectionError(
                f"Stage-C {language} selected base row changed: {sample_id!r}."
            )
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--fleurs-release-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--kk-ready-manifest", type=Path, required=True)
    parser.add_argument("--mixed-ready-manifest", type=Path, action="append", required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--ru-raw-manifest", type=Path, required=True)
    parser.add_argument("--ru-ready-manifest", type=Path, required=True)
    parser.add_argument("--ru-rejections", type=Path, required=True)
    parser.add_argument("--kk-output-manifest", type=Path, required=True)
    parser.add_argument("--mixed-output-manifest", type=Path, required=True)
    parser.add_argument("--combined-output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = (
        arguments.ru_raw_manifest,
        arguments.ru_ready_manifest,
        arguments.ru_rejections,
        arguments.kk_output_manifest,
        arguments.mixed_output_manifest,
        arguments.combined_output_manifest,
        arguments.output_receipt,
    )
    try:
        if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
            raise FreshSuiteSelectionError(
                "Stage-C materialization outputs must be distinct and write-once."
            )
        if any(not path.parent.is_dir() for path in outputs):
            raise FreshSuiteSelectionError(
                "Every Stage-C materialization output parent must exist."
            )
        if not arguments.slice_name.replace("-", "").replace("_", "").isalnum():
            raise FreshSuiteSelectionError("Stage-C slice-name is not portable.")
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        plan = load_fresh_suite_selection(arguments.selection, project_root)
        ru_items = _role_items(plan, "ru")
        kk_items = _role_items(plan, "kk")
        mixed_items = _role_items(plan, "mixed")

        _release, records = inspect_fleurs_release(arguments.fleurs_release_root, "ru_ru")
        records_by_member = {f"test/{record.filename}": record for record in records["test"]}
        ru_records = []
        for item in ru_items:
            member = cast(str, item["source_member"])
            record = records_by_member.get(member)
            if record is None or (
                f"google_fleurs_ru_v1:{record.filename.removesuffix('.wav')}"
                != item["sample_id"]
                or record.text_hash != item["text_hash"]
                or record.transcript != item["text"]
            ):
                raise FreshSuiteSelectionError(
                    f"Frozen RU selection is not bound to release member {member!r}."
                )
            ru_records.append(record)
        raw_destination = (
            data_root
            / "raw"
            / "google_fleurs_ru_v1"
            / "slices"
            / arguments.slice_name
        )
        raw_destination.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_fleurs_audio_slice(
            arguments.fleurs_release_root,
            "ru_ru",
            "test",
            ru_records,
            raw_destination,
        )
        assets: dict[str, FleursExtractedAsset] = {}
        for filename, path in extracted.items():
            duration, sample_rate, codec = inspect_extracted_fleurs_audio(path)
            assets[filename] = FleursExtractedAsset(
                filename=filename,
                relative_path=path.relative_to(data_root).as_posix(),
                sha256=sha256_file(path),
                duration_s=duration,
                original_sr=sample_rate,
                codec=codec,
            )
        ru_raw = fleurs_manifest_rows(
            ru_records, assets, manifest_split="test", created_at=arguments.created_at
        )
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest(ru_raw)
        validate_manifest_licenses(ru_raw, ledger)
        require_valid_assets(ru_raw, data_root)
        prepared = preprocess_rows(
            ru_raw, data_root, AudioPreparationPipeline(), allow_rejections=True
        )
        ru_ready = list(prepared.processed_rows)
        if not ru_ready:
            raise FreshSuiteSelectionError("Every selected RU asset failed QA; stopping safely.")

        kk_ready = _selected_ready_rows(
            kk_items, load_manifest(arguments.kk_ready_manifest), "kk"
        )
        mixed_ready = _selected_ready_rows(
            mixed_items, _load_manifests(arguments.mixed_ready_manifest), "mixed"
        )
        combined = sorted([*ru_ready, *kk_ready, *mixed_ready], key=lambda row: row.sample_id)
        for rows in (ru_ready, kk_ready, mixed_ready, combined):
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            require_valid_assets(rows, data_root)

        with tempfile.TemporaryDirectory(
            prefix="kds-stage-c-base-manifests-", dir=arguments.output_receipt.parent
        ) as stage_name:
            stage = Path(stage_name)
            staged = {path: stage / path.name for path in outputs}
            write_manifest(staged[arguments.ru_raw_manifest], ru_raw)
            write_manifest(staged[arguments.ru_ready_manifest], ru_ready)
            write_manifest(staged[arguments.kk_output_manifest], kk_ready)
            write_manifest(staged[arguments.mixed_output_manifest], mixed_ready)
            write_manifest(staged[arguments.combined_output_manifest], combined)
            staged[arguments.ru_rejections].write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selection_path": arguments.selection.as_posix(),
                        "selection_sha256": sha256_file(arguments.selection),
                        "selected_rows": len(ru_raw),
                        "ready_rows": len(ru_ready),
                        "rejected_rows": [
                            {
                                "sample_id": issue.sample_id,
                                "relative_path": issue.relative_path,
                                "reason": issue.detail,
                            }
                            for issue in prepared.issues
                        ],
                        "post_selection_backfill": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            output_manifests = {
                "ru_raw": arguments.ru_raw_manifest,
                "ru_ready": arguments.ru_ready_manifest,
                "kk_ready": arguments.kk_output_manifest,
                "mixed_ready": arguments.mixed_output_manifest,
                "combined_ready": arguments.combined_output_manifest,
                "ru_rejections": arguments.ru_rejections,
            }
            staged[arguments.output_receipt].write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": "fresh-suite-stage-c-base-materialization-v1",
                        "created_at": arguments.created_at,
                        "selection": {
                            "path": arguments.selection.as_posix(),
                            "sha256": sha256_file(arguments.selection),
                        },
                        "counts": {
                            "selected": {"ru": 55, "kk": 60, "mixed": 58},
                            "ready": {
                                "ru": len(ru_ready),
                                "kk": len(kk_ready),
                                "mixed": len(mixed_ready),
                            },
                            "rejected": {"ru": len(prepared.issues), "kk": 0, "mixed": 0},
                        },
                        "outputs": {
                            name: {
                                "path": path.as_posix(),
                                "sha256": sha256_file(staged[path]),
                            }
                            for name, path in output_manifests.items()
                        },
                        "post_selection_backfill": False,
                        "detector_inference_authorized": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in outputs):
                raise FreshSuiteSelectionError(
                    "A Stage-C materialization output appeared during staging."
                )
            for path in outputs:
                staged[path].replace(path)
    except (
        FleursIngestionError,
        FreshSuiteSelectionError,
        LicenseLedgerError,
        ManifestError,
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
                "ready": {"ru": len(ru_ready), "kk": len(kk_ready), "mixed": len(mixed_ready)},
                "ru_rejected": len(prepared.issues),
                "combined": len(combined),
                "receipt": str(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
