"""Publish the balanced Stage-C candidate after complete KazakhTTS rejection accounting."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.kazakhtts_candidate import KazakhTtsCandidateError, build_kazakhtts_pairs
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.eval.fresh_suite_selection import FreshSuiteSelectionError, load_fresh_suite_selection


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KazakhTtsCandidateError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise KazakhTtsCandidateError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_synthesis_bindings(
    report_path: Path,
    *,
    selection: Path,
    base_manifest: Path,
    raw_manifest: Path,
    text_rejections: Path,
    normalization_plan: Path,
) -> Mapping[str, object]:
    report = _json_object(report_path, "Stage-C synthesis report")
    bindings = (
        ("selection", selection),
        ("base_manifest", base_manifest),
        ("output_manifest", raw_manifest),
        ("text_rejections", text_rejections),
        ("normalization_plan", normalization_plan),
    )
    if (
        report.get("schema_version") != 1
        or report.get("protocol_id")
        != "fresh-suite-stage-c-kazakhtts-normalized-synthesis-v2"
        or report.get("selected_base_rows") != 168
        or report.get("generated_rows") != 168
        or report.get("text_rejected_rows") != 0
        or report.get("generated_by_language") != {"kk": 60, "mixed": 58, "ru": 50}
        or report.get("detector_inference_performed") is not False
        or report.get("full_asset_acoustic_gate_passed") is not False
    ):
        raise KazakhTtsCandidateError("Stage-C synthesis report state is invalid.")
    for name, path in bindings:
        binding = report.get(name)
        if (
            not isinstance(binding, dict)
            or binding.get("path") != path.as_posix()
            or binding.get("sha256") != sha256_file(path)
        ):
            raise KazakhTtsCandidateError(
                f"Stage-C synthesis report has an invalid {name} binding."
            )
    return report


def _rejection_sets(
    *,
    selection: Path,
    raw_manifest: Path,
    text_rejections_path: Path,
    audio_rejections_path: Path,
) -> tuple[set[str], set[str]]:
    text = _json_object(text_rejections_path, "Stage-C text rejection report")
    rejected_text_rows = text.get("rejected_text_rows")
    if (
        text.get("schema_version") != 1
        or text.get("selection_sha256") != sha256_file(selection)
        or text.get("selected_base_rows") != 168
        or text.get("accepted_text_rows") != 168
        or text.get("post_selection_backfill") is not False
        or not isinstance(rejected_text_rows, list)
    ):
        raise KazakhTtsCandidateError("Stage-C text rejection report is invalid.")
    text_ids: set[str] = set()
    for item in rejected_text_rows:
        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
            raise KazakhTtsCandidateError("Stage-C text rejection row is invalid.")
        text_ids.add(cast(str, item["sample_id"]))

    audio = _json_object(audio_rejections_path, "Stage-C audio rejection report")
    rejected_audio_rows = audio.get("rejected_rows")
    if (
        audio.get("input_manifest") != raw_manifest.as_posix()
        or audio.get("reused_rows") != 0
        or audio.get("published_rows") != 167
        or not isinstance(rejected_audio_rows, list)
    ):
        raise KazakhTtsCandidateError("Stage-C audio rejection report is invalid.")
    audio_ids: set[str] = set()
    for item in rejected_audio_rows:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sample_id"), str)
            or not isinstance(item.get("relative_path"), str)
            or not isinstance(item.get("detail"), str)
        ):
            raise KazakhTtsCandidateError("Stage-C audio rejection row is invalid.")
        audio_ids.add(cast(str, item["sample_id"]))
    if len(audio_ids) != len(rejected_audio_rows):
        raise KazakhTtsCandidateError("Stage-C audio rejection IDs are repeated.")
    return text_ids, audio_ids


def _validate_pairs(rows: list[ManifestRow]) -> None:
    expected = {
        "ru": Counter({"bonafide": 50, "spoof": 50}),
        "kk": Counter({"bonafide": 60, "spoof": 60}),
        "mixed": Counter({"bonafide": 57, "spoof": 57}),
    }
    for language, counts in expected.items():
        actual = Counter(row.label for row in rows if row.language == language)
        if actual != counts:
            raise KazakhTtsCandidateError(
                f"Stage-C {language} candidate counts changed: {dict(actual)}."
            )
    by_text: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_text.setdefault(row.text_id, []).append(row)
    if len(by_text) != 167:
        raise KazakhTtsCandidateError("Stage-C candidate must contain 167 unique text pairs.")
    for text_id, pair in by_text.items():
        if (
            len(pair) != 2
            or {row.label for row in pair} != {"bonafide", "spoof"}
            or len({row.text_hash for row in pair}) != 1
            or len({row.language for row in pair}) != 1
            or len({row.code_switch for row in pair}) != 1
        ):
            raise KazakhTtsCandidateError(f"Stage-C pair is invalid for {text_id!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--raw-spoof-manifest", type=Path, required=True)
    parser.add_argument("--ready-spoof-manifest", type=Path, required=True)
    parser.add_argument("--text-rejections", type=Path, required=True)
    parser.add_argument("--audio-rejections", type=Path, required=True)
    parser.add_argument("--synthesis-report", type=Path, required=True)
    parser.add_argument("--normalization-plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-ru", type=Path, required=True)
    parser.add_argument("--output-kk", type=Path, required=True)
    parser.add_argument("--output-mixed", type=Path, required=True)
    parser.add_argument("--output-combined", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    outputs = {
        "ru": arguments.output_ru,
        "kk": arguments.output_kk,
        "mixed": arguments.output_mixed,
        "combined": arguments.output_combined,
    }
    all_outputs = (*outputs.values(), arguments.output_receipt)
    try:
        if (
            len(set(all_outputs)) != len(all_outputs)
            or any(path.exists() or not path.parent.is_dir() for path in all_outputs)
        ):
            raise KazakhTtsCandidateError(
                "Stage-C candidate outputs must be distinct, new and have existing parents."
            )
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        load_fresh_suite_selection(
            arguments.selection, arguments.project_root.resolve(strict=True)
        )
        _require_synthesis_bindings(
            arguments.synthesis_report,
            selection=arguments.selection,
            base_manifest=arguments.base_manifest,
            raw_manifest=arguments.raw_spoof_manifest,
            text_rejections=arguments.text_rejections,
            normalization_plan=arguments.normalization_plan,
        )
        base = load_manifest(arguments.base_manifest)
        raw = load_manifest(arguments.raw_spoof_manifest)
        ready = load_manifest(arguments.ready_spoof_manifest)
        for rows in (base, raw, ready):
            validate_manifest(rows)
        ledger = load_license_ledger(arguments.license_ledger)
        for rows in (base, raw, ready):
            validate_manifest_licenses(rows, ledger)
            require_valid_assets(rows, arguments.data_root)
        text_ids, audio_ids = _rejection_sets(
            selection=arguments.selection,
            raw_manifest=arguments.raw_spoof_manifest,
            text_rejections_path=arguments.text_rejections,
            audio_rejections_path=arguments.audio_rejections,
        )
        pairs = build_kazakhtts_pairs(
            base_rows=base,
            raw_spoof_rows=raw,
            ready_spoof_rows=ready,
            text_rejected_base_ids=text_ids,
            rejected_spoof_ids=audio_ids,
        )
        pairs = sorted(pairs, key=lambda row: (row.language, row.text_id, row.label))
        validate_manifest(pairs)
        _validate_pairs(pairs)
        staged_root = Path(
            tempfile.mkdtemp(prefix=".kds-stage-c-candidate-", dir=arguments.output_receipt.parent)
        )
        try:
            staged: dict[str, Path] = {}
            for name, output in outputs.items():
                staged_path = staged_root / output.name
                rows = (
                    pairs
                    if name == "combined"
                    else [row for row in pairs if row.language == name]
                )
                write_manifest(staged_path, rows)
                staged[name] = staged_path
            receipt_path = staged_root / arguments.output_receipt.name
            receipt = {
                "schema_version": 1,
                "protocol_id": "fresh-suite-stage-c-kazakhtts-pairing-v1",
                "created_at": arguments.created_at,
                "inputs": {
                    name: {"path": path.as_posix(), "sha256": sha256_file(path)}
                    for name, path in {
                        "selection": arguments.selection,
                        "base_manifest": arguments.base_manifest,
                        "raw_spoof_manifest": arguments.raw_spoof_manifest,
                        "ready_spoof_manifest": arguments.ready_spoof_manifest,
                        "text_rejections": arguments.text_rejections,
                        "audio_rejections": arguments.audio_rejections,
                        "synthesis_report": arguments.synthesis_report,
                        "normalization_plan": arguments.normalization_plan,
                        "license_ledger": arguments.license_ledger,
                    }.items()
                },
                "counts": {
                    "base_ready": len(base),
                    "raw_spoof": len(raw),
                    "ready_spoof": len(ready),
                    "text_rejected": len(text_ids),
                    "audio_rejected": len(audio_ids),
                    "pairs": len(pairs) // 2,
                    "candidate_assets": len(pairs),
                    "pairs_by_language": {"kk": 60, "mixed": 57, "ru": 50},
                },
                "outputs": {
                    name: {
                        "path": outputs[name].as_posix(),
                        "sha256": sha256_file(staged_path),
                        "rows": len(load_manifest(staged_path)),
                    }
                    for name, staged_path in staged.items()
                },
                "decision_rule": {
                    "post_selection_backfill": False,
                    "metric_or_detector_based_selection": False,
                    "metrics_must_remain_separate_by_language": True,
                    "source_independent": False,
                    "speaker_independent": False,
                },
                "full_asset_acoustic_gate_passed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if any(path.exists() for path in all_outputs):
                raise KazakhTtsCandidateError(
                    "A Stage-C candidate output appeared while staging."
                )
            for name, output in outputs.items():
                staged[name].replace(output)
            receipt_path.replace(arguments.output_receipt)
        finally:
            shutil.rmtree(staged_root, ignore_errors=True)
    except (
        FreshSuiteSelectionError,
        KazakhTtsCandidateError,
        LicenseLedgerError,
        ManifestError,
        OSError,
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
                "pairs": len(pairs) // 2,
                "pairs_by_language": {"kk": 60, "mixed": 57, "ru": 50},
                "output": str(arguments.output_combined),
                "receipt": str(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
