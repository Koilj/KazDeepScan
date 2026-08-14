"""Run resumable v4 source decode/QA and historical perceptual-fingerprint journals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from kds.data.assets import require_valid_assets, resolve_asset_path, sha256_file
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestRow, load_manifest, validate_manifest, write_manifest
from kds.data.v4_audio_gate import (
    V4_AUDIO_FINGERPRINT_VERSION,
    V4_NEAR_AUDIO_HAMMING_THRESHOLD,
    V4_NEAR_AUDIO_SPEECH_DURATION_RATIO,
    V4AudioGateError,
    V4AudioSignature,
    V4DecodedCandidate,
    V4DecodedDecision,
    V4DecodeResult,
    V4DecodeTask,
    V4HistoryFingerprintResult,
    V4HistoryFingerprintTask,
    V4NearAudioMatch,
    append_v4_decode_journal,
    append_v4_history_fingerprint_journal,
    decide_v4_decoded_audio_eligibility,
    decode_tasks_by_id,
    decoded_relative_path,
    load_v4_decode_journal,
    load_v4_history_fingerprint_journal,
    run_v4_decode_task,
    run_v4_history_fingerprint_task,
)
from kds.data.v4_materialization import V4_RAW_INVENTORY_FIELDS


@dataclass(frozen=True, slots=True)
class HistoricalExposure:
    exact_references: dict[str, tuple[str, ...]]
    fingerprint_tasks: dict[str, V4HistoryFingerprintTask]
    unique_audio_hashes: int
    available_audio_hashes: int
    unavailable_audio_hashes: int


V4_DECODE_INVENTORY_FIELDS = (
    "selection_rank",
    "target_state",
    "language",
    "label",
    "candidate_id",
    "pair_id",
    "source_id",
    "source_lineage_id",
    "source_component",
    "parent_group_id",
    "canonical_text_hash",
    "raw_relative_path",
    "raw_audio_sha256",
    "decoded_relative_path",
    "decoded_audio_sha256",
    "decoded_size_bytes",
    "duration_s",
    "peak",
    "rms_dbfs",
    "clipped_fraction",
    "dc_offset",
    "speech_seconds",
    "speech_segment_count",
    "audio_fingerprint_v1",
    "preparation_status",
    "eligibility_status",
    "rejection_reason",
    "exact_duplicate_of_candidate_id",
    "historical_exact_match_count",
    "historical_exact_matches",
    "historical_near_match_count",
    "historical_near_matches",
    "within_pool_near_match_count",
    "within_pool_near_matches",
    "frozen_source_train_state",
)

V4_HISTORY_FINGERPRINT_FIELDS = (
    "manifest_audio_sha256",
    "reference_count",
    "references",
    "local_asset_state",
    "selected_relative_path",
    "canonical_audio_sha256",
    "duration_s",
    "speech_seconds",
    "audio_fingerprint_v1",
    "fingerprint_status",
    "detail",
)


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V4AudioGateError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4AudioGateError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4AudioGateError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise V4AudioGateError(f"{label} must be a JSON list.")
    return cast(list[object], value)


def _load_raw_inventory(path: Path) -> dict[str, dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != V4_RAW_INVENTORY_FIELDS:
                raise V4AudioGateError("v4 raw inventory schema is invalid.")
            rows = list(reader)
    except OSError as error:
        raise V4AudioGateError("Cannot read v4 raw inventory.") from error
    eligible = {
        row["candidate_id"]: row
        for row in rows
        if row["eligibility_status"] == "eligible_for_decode_qa"
    }
    if len(rows) != 21_600 or len(eligible) != 21_598:
        raise V4AudioGateError("v4 raw inventory accounting changed.")
    return eligible


def _build_decode_tasks(
    raw_rows: list[ManifestRow],
    inventory: dict[str, dict[str, str]],
    data_root: Path,
) -> dict[str, V4DecodeTask]:
    tasks: list[V4DecodeTask] = []
    for row in raw_rows:
        context = inventory.get(row.sample_id)
        relative = decoded_relative_path(row.sha256)
        if (
            context is None
            or context["raw_audio_sha256"] != row.sha256
            or context["raw_relative_path"] != row.relative_path
        ):
            raise V4AudioGateError("v4 raw manifest and inventory are inconsistent.")
        tasks.append(
            V4DecodeTask(
                sample_id=row.sample_id,
                raw_relative_path=row.relative_path,
                raw_sha256=row.sha256,
                source_path=str(resolve_asset_path(data_root, row.relative_path)),
                decoded_relative_path=relative,
                destination_path=str(resolve_asset_path(data_root, relative)),
            )
        )
    if len(tasks) != len(inventory):
        raise V4AudioGateError("v4 raw manifest does not cover the eligible inventory.")
    return decode_tasks_by_id(tasks)


def _load_historical_exposure(
    source_receipt: dict[str, object],
    project_root: Path,
    data_root: Path,
) -> HistoricalExposure:
    bindings = _mapping(source_receipt.get("bindings"), "source materialization bindings")
    exposure = _mapping(bindings.get("historical_exposure"), "historical exposure")
    manifest_bindings = _sequence(exposure.get("manifests"), "historical manifests")
    if exposure.get("manifest_count") != len(manifest_bindings) or len(manifest_bindings) != 99:
        raise V4AudioGateError("Historical manifest binding count changed.")
    references: dict[str, list[str]] = {}
    rows_by_hash: dict[str, list[ManifestRow]] = {}
    for raw_binding in manifest_bindings:
        binding = _mapping(raw_binding, "historical manifest binding")
        relative = binding.get("path")
        if not isinstance(relative, str):
            raise V4AudioGateError("Historical manifest path is invalid.")
        path = (project_root / relative).resolve(strict=True)
        path.relative_to(project_root)
        rows = load_manifest(path)
        if sha256_file(path) != binding.get("sha256") or len(rows) != binding.get("rows"):
            raise V4AudioGateError(f"Historical manifest binding changed: {relative}")
        for row in rows:
            reference = f"{relative}:{row.sample_id}"
            references.setdefault(row.sha256, []).append(reference)
            rows_by_hash.setdefault(row.sha256, []).append(row)
    tasks: dict[str, V4HistoryFingerprintTask] = {}
    for digest, rows in rows_by_hash.items():
        available: list[tuple[int, str, Path]] = []
        for row in rows:
            path = resolve_asset_path(data_root, row.relative_path)
            if path.is_file():
                preferred = int(not (row.codec == "wav" and row.original_sr == 16_000))
                available.append((preferred, row.relative_path, path))
        if not available:
            continue
        _preferred, relative_path, source_path = min(available)
        task = V4HistoryFingerprintTask(
            identity=f"history:{digest}",
            manifest_audio_sha256=digest,
            relative_path=relative_path,
            source_path=str(source_path),
        )
        tasks[task.identity] = task
    exact_references = {digest: tuple(sorted(items)) for digest, items in references.items()}
    return HistoricalExposure(
        exact_references=exact_references,
        fingerprint_tasks=tasks,
        unique_audio_hashes=len(rows_by_hash),
        available_audio_hashes=len(tasks),
        unavailable_audio_hashes=len(rows_by_hash) - len(tasks),
    )


def _ensure_runtime_binding(path: Path, binding: dict[str, object]) -> None:
    serialized = json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise V4AudioGateError("Existing v4 decode runtime binding differs.")
        return
    path.write_text(serialized, encoding="utf-8")


def _run_decode_workers(
    tasks: dict[str, V4DecodeTask],
    results: dict[str, V4DecodeResult],
    journal: Path,
    workers: int,
) -> None:
    remaining = [task for sample_id, task in tasks.items() if sample_id not in results]
    if not remaining:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[V4DecodeResult], V4DecodeTask] = {
            executor.submit(run_v4_decode_task, task): task for task in remaining
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_v4_decode_journal(journal, result)
            results[result.sample_id] = result
            if completed % 100 == 0 or completed == len(remaining):
                _progress("source_decode_qa", len(results), len(tasks))


def _run_history_workers(
    tasks: dict[str, V4HistoryFingerprintTask],
    results: dict[str, V4HistoryFingerprintResult],
    journal: Path,
    workers: int,
) -> None:
    remaining = [task for identity, task in tasks.items() if identity not in results]
    if not remaining:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[V4HistoryFingerprintResult], V4HistoryFingerprintTask] = {
            executor.submit(run_v4_history_fingerprint_task, task): task for task in remaining
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_v4_history_fingerprint_journal(journal, result)
            results[result.identity] = result
            if completed % 100 == 0 or completed == len(remaining):
                _progress("historical_audio_fingerprints", len(results), len(tasks))


def _progress(stage: str, completed: int, total: int) -> None:
    print(
        json.dumps(
            {"status": "progress", "stage": stage, "completed": completed, "total": total},
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _decoded_decisions(
    inventory: dict[str, dict[str, str]],
    decode_results: dict[str, V4DecodeResult],
    history: HistoricalExposure,
    history_results: dict[str, V4HistoryFingerprintResult],
) -> tuple[V4DecodedDecision, ...]:
    exact_references = {digest: list(items) for digest, items in history.exact_references.items()}
    canonical_history: dict[str, V4HistoryFingerprintResult] = {}
    for result in history_results.values():
        if result.status != "fingerprinted":
            continue
        exact_references.setdefault(result.canonical_audio_sha256, []).append(result.identity)
        canonical_history.setdefault(result.canonical_audio_sha256, result)
    historical_signatures = tuple(
        V4AudioSignature(
            identity=f"history-canonical:{digest}",
            audio_sha256=digest,
            fingerprint=result.audio_fingerprint_v1,
            speech_seconds=result.speech_seconds,
        )
        for digest, result in sorted(canonical_history.items())
    )
    candidates = tuple(
        V4DecodedCandidate(
            selection_rank=int(inventory[sample_id]["selection_rank"]),
            language=inventory[sample_id]["language"],
            label=inventory[sample_id]["label"],
            result=result,
        )
        for sample_id, result in decode_results.items()
    )
    return decide_v4_decoded_audio_eligibility(
        candidates,
        exact_references,
        historical_signatures,
    )


def _ready_manifest_rows(
    decisions: tuple[V4DecodedDecision, ...],
    raw_rows: list[ManifestRow],
    created_at: str,
) -> tuple[ManifestRow, ...]:
    raw_by_id = {row.sample_id: row for row in raw_rows}
    rows = []
    for decision in decisions:
        if decision.eligibility_status != "eligible":
            continue
        result = decision.candidate.result
        rows.append(
            replace(
                raw_by_id[result.sample_id],
                relative_path=result.decoded_relative_path,
                sha256=result.decoded_audio_sha256,
                duration_s=result.duration_s,
                original_sr=16_000,
                codec="wav",
                created_at=created_at,
            )
        )
    validate_manifest(rows)
    return tuple(rows)


def _freeze_source_train_rows(
    decisions: tuple[V4DecodedDecision, ...],
    ready_rows: tuple[ManifestRow, ...],
    *,
    target_per_cell: int = 5_000,
) -> tuple[ManifestRow, ...]:
    ready_by_id = {row.sample_id: row for row in ready_rows}
    selected: list[ManifestRow] = []
    for cell in ("kk/bonafide", "ru/bonafide", "ru/spoof"):
        eligible = sorted(
            (
                item
                for item in decisions
                if item.eligibility_status == "eligible" and item.candidate.cell == cell
            ),
            key=lambda item: (
                item.candidate.selection_rank,
                item.candidate.result.sample_id,
            ),
        )
        if len(eligible) < target_per_cell:
            raise V4AudioGateError(
                f"v4 source cell {cell} has {len(eligible)} eligible rows; "
                f"needs {target_per_cell}."
            )
        selected.extend(
            ready_by_id[item.candidate.result.sample_id]
            for item in eligible[:target_per_cell]
        )
    validate_manifest(selected)
    return tuple(selected)


def _match_json(matches: Sequence[V4NearAudioMatch]) -> str:
    return json.dumps([asdict(match) for match in matches], sort_keys=True, separators=(",", ":"))


def _publish_results(
    *,
    project_root: Path,
    data_root: Path,
    raw_rows: list[ManifestRow],
    inventory: dict[str, dict[str, str]],
    decisions: tuple[V4DecodedDecision, ...],
    history: HistoricalExposure,
    history_results: dict[str, V4HistoryFingerprintResult],
    decode_journal: Path,
    history_journal: Path,
    source_receipt_path: Path,
    license_ledger_path: Path,
    output_decode_inventory: Path,
    output_history_inventory: Path,
    output_ready_manifest: Path,
    output_frozen_manifest: Path,
    output_receipt: Path,
    created_at: str,
) -> None:
    outputs = (
        output_decode_inventory,
        output_history_inventory,
        output_ready_manifest,
        output_frozen_manifest,
        output_receipt,
    )
    if len(set(outputs)) != len(outputs) or any(
        path.exists() or not path.parent.is_dir() for path in outputs
    ):
        raise V4AudioGateError("v4 decode/QA publication outputs must be distinct and new.")
    ready_rows = _ready_manifest_rows(decisions, raw_rows, created_at)
    frozen_rows = _freeze_source_train_rows(decisions, ready_rows)
    ledger = load_license_ledger(license_ledger_path)
    validate_manifest_licenses(ready_rows, ledger)
    validate_manifest_licenses(frozen_rows, ledger)
    require_valid_assets(ready_rows, data_root)
    frozen_ids = {row.sample_id for row in frozen_rows}
    decision_by_id = {item.candidate.result.sample_id: item for item in decisions}
    published: list[tuple[Path, Path]] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-decode-publication-", dir=project_root
        ) as stage_name:
            stage = Path(stage_name)
            staged_decode = stage / output_decode_inventory.name
            staged_history = stage / output_history_inventory.name
            staged_ready = stage / output_ready_manifest.name
            staged_frozen = stage / output_frozen_manifest.name
            staged_receipt = stage / output_receipt.name
            with staged_decode.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=V4_DECODE_INVENTORY_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for sample_id, context in sorted(
                    inventory.items(),
                    key=lambda item: (
                        item[1]["language"],
                        item[1]["label"],
                        int(item[1]["selection_rank"]),
                        item[0],
                    ),
                ):
                    decision = decision_by_id[sample_id]
                    result = decision.candidate.result
                    writer.writerow(
                        {
                            **{
                                field: context[field]
                                for field in V4_DECODE_INVENTORY_FIELDS
                                if field in context
                            },
                            "decoded_relative_path": result.decoded_relative_path,
                            "decoded_audio_sha256": result.decoded_audio_sha256,
                            "decoded_size_bytes": result.decoded_size_bytes,
                            "duration_s": result.duration_s,
                            "peak": result.peak,
                            "rms_dbfs": result.rms_dbfs,
                            "clipped_fraction": result.clipped_fraction,
                            "dc_offset": result.dc_offset,
                            "speech_seconds": result.speech_seconds,
                            "speech_segment_count": result.speech_segment_count,
                            "audio_fingerprint_v1": result.audio_fingerprint_v1,
                            "preparation_status": result.preparation_status,
                            "eligibility_status": decision.eligibility_status,
                            "rejection_reason": decision.rejection_reason,
                            "exact_duplicate_of_candidate_id": (
                                decision.exact_duplicate_of_candidate_id
                            ),
                            "historical_exact_match_count": len(decision.historical_exact_matches),
                            "historical_exact_matches": json.dumps(
                                decision.historical_exact_matches
                            ),
                            "historical_near_match_count": len(decision.historical_near_matches),
                            "historical_near_matches": _match_json(
                                decision.historical_near_matches
                            ),
                            "within_pool_near_match_count": len(decision.within_pool_near_matches),
                            "within_pool_near_matches": _match_json(
                                decision.within_pool_near_matches
                            ),
                            "frozen_source_train_state": (
                                "selected" if sample_id in frozen_ids else "not_selected"
                            ),
                        }
                    )
            history_by_manifest_hash = {
                item.manifest_audio_sha256: item for item in history_results.values()
            }
            with staged_history.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=V4_HISTORY_FINGERPRINT_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for digest, references in sorted(history.exact_references.items()):
                    history_result = history_by_manifest_hash.get(digest)
                    writer.writerow(
                        {
                            "manifest_audio_sha256": digest,
                            "reference_count": len(references),
                            "references": json.dumps(references, ensure_ascii=False),
                            "local_asset_state": (
                                "available" if history_result else "unavailable"
                            ),
                            "selected_relative_path": (
                                history_result.relative_path if history_result else ""
                            ),
                            "canonical_audio_sha256": (
                                history_result.canonical_audio_sha256
                                if history_result
                                else ""
                            ),
                            "duration_s": history_result.duration_s if history_result else "",
                            "speech_seconds": (
                                history_result.speech_seconds if history_result else ""
                            ),
                            "audio_fingerprint_v1": (
                                history_result.audio_fingerprint_v1 if history_result else ""
                            ),
                            "fingerprint_status": (
                                history_result.status if history_result else "not_fingerprinted"
                            ),
                            "detail": (
                                history_result.detail
                                if history_result
                                else "local_asset_unavailable"
                            ),
                        }
                    )
            write_manifest(staged_ready, ready_rows)
            write_manifest(staged_frozen, frozen_rows)
            decision_counts = Counter(
                (item.candidate.cell, item.eligibility_status, item.rejection_reason)
                for item in decisions
            )
            ready_counts = Counter(
                f"{item.candidate.language}/{item.candidate.label}"
                for item in decisions
                if item.eligibility_status == "eligible"
            )
            receipt = {
                "schema_version": 1,
                "protocol_id": "xlsr-sls-model-v4-source-decode-qa-v1",
                "created_at": created_at,
                "state": "source_train_frozen_15000_kk_spoof_synthesis_authorized",
                "bindings": {
                    "source_materialization_receipt": {
                        "path": source_receipt_path.as_posix(),
                        "sha256": sha256_file(source_receipt_path),
                    },
                    "license_ledger": {
                        "path": license_ledger_path.as_posix(),
                        "sha256": sha256_file(license_ledger_path),
                    },
                    "decode_journal": {
                        "path": decode_journal.relative_to(project_root).as_posix(),
                        "sha256": sha256_file(decode_journal),
                        "rows": len(decisions),
                    },
                    "historical_fingerprint_journal": {
                        "path": history_journal.relative_to(project_root).as_posix(),
                        "sha256": sha256_file(history_journal),
                        "rows": len(history_results),
                    },
                },
                "audio_contract": {
                    "decode": "ffmpeg mono pcm_s16le 16000 Hz",
                    "qa": "QualityPolicy defaults plus WebRTC VAD defaults",
                    "minimum_speech_seconds": 2.5,
                    "fingerprint_version": V4_AUDIO_FINGERPRINT_VERSION,
                    "near_audio_hamming_threshold": V4_NEAR_AUDIO_HAMMING_THRESHOLD,
                    "near_audio_speech_duration_ratio": V4_NEAR_AUDIO_SPEECH_DURATION_RATIO,
                },
                "outputs": {
                    "decode_inventory": {
                        "path": output_decode_inventory.as_posix(),
                        "sha256": sha256_file(staged_decode),
                        "rows": len(decisions),
                    },
                    "historical_fingerprint_inventory": {
                        "path": output_history_inventory.as_posix(),
                        "sha256": sha256_file(staged_history),
                        "rows": history.unique_audio_hashes,
                    },
                    "ready_manifest": {
                        "path": output_ready_manifest.as_posix(),
                        "sha256": sha256_file(staged_ready),
                        "rows": len(ready_rows),
                    },
                    "frozen_source_train_manifest": {
                        "path": output_frozen_manifest.as_posix(),
                        "sha256": sha256_file(staged_frozen),
                        "rows": len(frozen_rows),
                    },
                },
                "accounting": {
                    "source_rows": len(decisions),
                    "ready_eligible_rows": len(ready_rows),
                    "ready_cell_counts": dict(sorted(ready_counts.items())),
                    "decision_counts": {
                        "|".join(key): value for key, value in sorted(decision_counts.items())
                    },
                    "historical_unique_manifest_hashes": history.unique_audio_hashes,
                    "historical_fingerprinted_hashes": history.available_audio_hashes,
                    "historical_unavailable_hashes": history.unavailable_audio_hashes,
                    "historical_unavailable_scope": "ML-DF Italian OOD only",
                    "historical_exact_decoded_collisions": sum(
                        bool(item.historical_exact_matches) for item in decisions
                    ),
                    "within_pool_exact_decoded_duplicates": sum(
                        bool(item.exact_duplicate_of_candidate_id) for item in decisions
                    ),
                    "historical_near_review_candidates_excluded": sum(
                        bool(item.historical_near_matches) for item in decisions
                    ),
                    "within_pool_near_review_candidates_excluded": sum(
                        bool(item.within_pool_near_matches) for item in decisions
                    ),
                },
                "balanced_train_decision": {
                    "previous_preferred_target": 24_000,
                    "actual_source_bottleneck_cell": "ru/bonafide",
                    "actual_source_bottleneck_eligible": ready_counts["ru/bonafide"],
                    "decision": "proceed_20k_balanced",
                    "target_per_language_label_cell": 5_000,
                    "frozen_source_cells": {
                        "kk/bonafide": 5_000,
                        "ru/bonafide": 5_000,
                        "ru/spoof": 5_000,
                    },
                    "pending_kk_spoof_target": 5_000,
                    "final_train_target": 20_000,
                    "replacement_or_backfill": False,
                },
                "claims": {
                    "technical_decode_qa_vad_performed": True,
                    "decoded_exact_audio_screen_performed": True,
                    "near_audio_screen_performed": True,
                    "near_audio_hits_excluded_from_frozen_source_train": True,
                    "speaker_independence": "not_verified_speaker_independent",
                    "kk_spoof_synthesis_authorized": True,
                    "training_authorized": False,
                    "new_dataset_search_performed": False,
                },
                "next_gate": (
                    "synthesize the frozen canonical v2 KK spoof candidates through Piper, MMS, "
                    "KazEmoTTS and Spark-TTS in new v4 namespaces; run the same decode/QA/VAD "
                    "and leakage gates; freeze exactly 5000 eligible rows"
                ),
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for staged, output in (
                (staged_decode, output_decode_inventory),
                (staged_history, output_history_inventory),
                (staged_ready, output_ready_manifest),
                (staged_frozen, output_frozen_manifest),
                (staged_receipt, output_receipt),
            ):
                os.link(staged, output)
                published.append((output, staged))
    except (OSError, ValueError) as error:
        for output, staged in reversed(published):
            try:
                if output.samefile(staged):
                    output.unlink()
            except OSError:
                pass
        raise V4AudioGateError(f"Cannot publish v4 decode/QA packet: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-inventory", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output-decode-inventory", type=Path, required=True)
    parser.add_argument("--output-history-inventory", type=Path, required=True)
    parser.add_argument("--output-ready-manifest", type=Path, required=True)
    parser.add_argument("--output-frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        if arguments.workers not in range(1, 65):
            raise V4AudioGateError("workers must be between 1 and 64.")
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        runtime = arguments.runtime_directory.resolve()
        runtime.relative_to(project_root / "artifacts")
        runtime.mkdir(parents=True, exist_ok=True)
        source_receipt = _json_object(arguments.source_receipt, "source materialization receipt")
        outputs = _mapping(source_receipt.get("outputs"), "source materialization outputs")
        raw_manifest_binding = _mapping(outputs.get("raw_manifest"), "raw manifest binding")
        if (
            source_receipt.get("state") != "source_raw_materialized_decode_qa_pending"
            or raw_manifest_binding.get("path") != arguments.raw_manifest.as_posix()
            or raw_manifest_binding.get("sha256") != sha256_file(arguments.raw_manifest)
        ):
            raise V4AudioGateError("Source materialization receipt binding failed.")
        raw_rows = load_manifest(arguments.raw_manifest)
        if len(raw_rows) != raw_manifest_binding.get("rows"):
            raise V4AudioGateError("Source raw manifest row count changed.")
        require_valid_assets(raw_rows, data_root)
        inventory = _load_raw_inventory(arguments.raw_inventory)
        tasks = _build_decode_tasks(raw_rows, inventory, data_root)
        history = _load_historical_exposure(source_receipt, project_root, data_root)
        binding = {
            "schema_version": 1,
            "protocol_id": "xlsr-sls-model-v4-source-decode-qa-runtime-v1",
            "source_receipt": {
                "path": arguments.source_receipt.as_posix(),
                "sha256": sha256_file(arguments.source_receipt),
            },
            "raw_manifest": {
                "path": arguments.raw_manifest.as_posix(),
                "sha256": sha256_file(arguments.raw_manifest),
                "rows": len(raw_rows),
            },
            "raw_inventory": {
                "path": arguments.raw_inventory.as_posix(),
                "sha256": sha256_file(arguments.raw_inventory),
                "rows": len(inventory),
            },
            "audio_contract": {
                "decode": "ffmpeg mono pcm_s16le 16000 Hz",
                "qa": "QualityPolicy defaults plus WebRTC VAD defaults",
                "minimum_speech_seconds": 2.5,
                "fingerprint_version": V4_AUDIO_FINGERPRINT_VERSION,
                "near_audio_hamming_threshold": V4_NEAR_AUDIO_HAMMING_THRESHOLD,
                "near_audio_speech_duration_ratio": V4_NEAR_AUDIO_SPEECH_DURATION_RATIO,
            },
            "historical_exposure": asdict(history)
            | {"fingerprint_tasks": len(history.fingerprint_tasks)},
        }
        # Exact references can be large and are already hash-bound through the 99 manifests.
        cast(dict[str, object], binding["historical_exposure"]).pop("exact_references")
        _ensure_runtime_binding(runtime / "binding.json", binding)
        decode_journal = runtime / "source_decode_qa.jsonl"
        decode_results = load_v4_decode_journal(decode_journal, tasks)
        _run_decode_workers(tasks, decode_results, decode_journal, arguments.workers)
        history_journal = runtime / "historical_audio_fingerprints.jsonl"
        history_results = load_v4_history_fingerprint_journal(
            history_journal, history.fingerprint_tasks
        )
        _run_history_workers(
            history.fingerprint_tasks,
            history_results,
            history_journal,
            arguments.workers,
        )
        if len(decode_results) != len(tasks) or len(history_results) != len(
            history.fingerprint_tasks
        ):
            raise V4AudioGateError("v4 decode/fingerprint journals are incomplete.")
        decisions = _decoded_decisions(inventory, decode_results, history, history_results)
        _publish_results(
            project_root=project_root,
            data_root=data_root,
            raw_rows=raw_rows,
            inventory=inventory,
            decisions=decisions,
            history=history,
            history_results=history_results,
            decode_journal=decode_journal,
            history_journal=history_journal,
            source_receipt_path=arguments.source_receipt,
            license_ledger_path=arguments.license_ledger,
            output_decode_inventory=arguments.output_decode_inventory,
            output_history_inventory=arguments.output_history_inventory,
            output_ready_manifest=arguments.output_ready_manifest,
            output_frozen_manifest=arguments.output_frozen_manifest,
            output_receipt=arguments.output_receipt,
            created_at=arguments.created_at,
        )
    except (OSError, ValueError, V4AudioGateError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "source_decode_results": len(decode_results),
                "source_status_counts": dict(
                    sorted(
                        Counter(
                            item.preparation_status for item in decode_results.values()
                        ).items()
                    )
                ),
                "historical_unique_audio_hashes": history.unique_audio_hashes,
                "historical_available_audio_hashes": history.available_audio_hashes,
                "historical_fingerprint_status_counts": dict(
                    sorted(Counter(item.status for item in history_results.values()).items())
                ),
                "runtime_directory": arguments.runtime_directory.as_posix(),
                "receipt": arguments.output_receipt.as_posix(),
                "receipt_sha256": sha256_file(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
