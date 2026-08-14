"""Run the frozen model-v4 KK spoof decode, QA/VAD and audio-leakage gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import require_valid_assets, resolve_asset_path, sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
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
    V4NearAudioMatch,
    append_v4_decode_journal,
    decide_v4_decoded_audio_eligibility,
    load_v4_decode_journal,
    run_v4_decode_task,
)

PROTOCOL_ID = "xlsr-sls-model-v4-kk-spoof-audio-gate-v1"
PLAN_SCHEMA_VERSION = 1
PROCESSED_ROOT = "processed/v4/xlsr_sls_model_v4_kk_spoof_decode_qa_v1"
RUNTIME_ROOT = "artifacts/v4/xlsr_sls_model_v4_kk_spoof_decode_qa_v1"
TARGET_PER_ROUTE = 1_250
ATTEMPTED_PER_ROUTE = 1_800

SYNTHESIS_INVENTORY_FIELDS = (
    "selection_rank",
    "target_state",
    "candidate_id",
    "pair_id",
    "generator_route_id",
    "generator_family",
    "model_id",
    "assigned_voice_id",
    "assigned_speaker_id",
    "assigned_emotion_id",
    "terminal_state",
    "actual_voice_id",
    "actual_speaker_id",
    "actual_emotion_id",
    "actual_seed",
    "generation_attempts",
    "retry_errors",
    "raw_relative_path",
    "raw_audio_sha256",
    "duration_s",
    "original_sr",
    "device",
    "error",
)

SOURCE_DECODE_INVENTORY_FIELDS = (
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

HISTORY_INVENTORY_FIELDS = (
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

OUTPUT_INVENTORY_FIELDS = (
    "selection_rank",
    "target_state",
    "candidate_id",
    "pair_id",
    "generator_route_id",
    "generator_family",
    "model_id",
    "assigned_voice_id",
    "actual_voice_id",
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
    "frozen_kk_spoof_train_state",
)


@dataclass(frozen=True, slots=True)
class Binding:
    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class Route:
    route_id: str
    raw_manifest: Binding
    synthesis_inventory: Binding
    synthesis_receipt: Binding


@dataclass(frozen=True, slots=True)
class Plan:
    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, Binding]
    routes: tuple[Route, ...]
    processed_root: str
    runtime_root: str
    output_inventory: str
    output_ready_manifest: str
    output_frozen_manifest: str
    output_receipt: str
    target_per_route: int


@dataclass(frozen=True, slots=True)
class CandidateContext:
    selection_rank: int
    target_state: str
    candidate_id: str
    pair_id: str
    route_id: str
    generator_family: str
    model_id: str
    assigned_voice_id: str
    actual_voice_id: str
    raw_relative_path: str
    raw_audio_sha256: str


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4AudioGateError(f"{label} must be a non-empty project-relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value == ".":
        raise V4AudioGateError(f"{label} must be a safe project-relative path.")
    return path.as_posix()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4AudioGateError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _load_binding(value: object, label: str, project_root: Path) -> Binding:
    raw = _mapping(value, label)
    if set(raw) != {"path", "sha256", "rows"}:
        raise V4AudioGateError(f"{label} keys are invalid.")
    path = _safe_path(raw["path"], f"{label} path")
    digest = _require_sha256(raw["sha256"], f"{label} SHA-256")
    rows = raw["rows"]
    if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0):
        raise V4AudioGateError(f"{label} rows must be null or a positive integer.")
    target = (project_root / path).resolve(strict=True)
    try:
        target.relative_to(project_root)
    except ValueError as error:
        raise V4AudioGateError(f"{label} escapes the project root.") from error
    if sha256_file(target) != digest:
        raise V4AudioGateError(f"{label} SHA-256 changed after the gate was frozen.")
    return Binding(path=path, sha256=digest, rows=rows)


def _load_route(value: object, position: int, project_root: Path) -> Route:
    raw = _mapping(value, f"route {position}")
    if set(raw) != {"route_id", "raw_manifest", "synthesis_inventory", "synthesis_receipt"}:
        raise V4AudioGateError(f"route {position} keys are invalid.")
    route_id = raw["route_id"]
    if not isinstance(route_id, str) or not route_id:
        raise V4AudioGateError(f"route {position} ID is invalid.")
    return Route(
        route_id=route_id,
        raw_manifest=_load_binding(
            raw["raw_manifest"], f"route {route_id} raw manifest", project_root
        ),
        synthesis_inventory=_load_binding(
            raw["synthesis_inventory"], f"route {route_id} synthesis inventory", project_root
        ),
        synthesis_receipt=_load_binding(
            raw["synthesis_receipt"], f"route {route_id} synthesis receipt", project_root
        ),
    )


def load_plan(path: Path, project_root: Path) -> Plan:
    raw = _json_object(path, "KK spoof audio gate plan")
    if set(raw) != {
        "schema_version",
        "protocol_id",
        "created_at",
        "state",
        "inputs",
        "routes",
        "outputs",
        "selection",
        "prohibitions",
    }:
        raise V4AudioGateError("KK spoof audio gate plan keys are invalid.")
    if (
        raw["schema_version"] != PLAN_SCHEMA_VERSION
        or raw["protocol_id"] != PROTOCOL_ID
        or raw["state"] != "frozen_pre_qa"
    ):
        raise V4AudioGateError("KK spoof audio gate plan state or version is invalid.")
    created_at = raw["created_at"]
    if not isinstance(created_at, str):
        raise V4AudioGateError("KK spoof audio gate plan timestamp is invalid.")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4AudioGateError("KK spoof audio gate plan timestamp is invalid.") from error
    inputs_raw = _mapping(raw["inputs"], "KK spoof audio gate plan inputs")
    required_inputs = {
        "source_decode_receipt",
        "source_decode_inventory",
        "historical_fingerprint_inventory",
        "license_ledger",
        "synthesis_plan",
        "audio_gate_module",
        "runner_script",
    }
    if set(inputs_raw) != required_inputs:
        raise V4AudioGateError("KK spoof audio gate input set is invalid.")
    inputs = {
        name: _load_binding(inputs_raw[name], f"audio gate input {name}", project_root)
        for name in sorted(required_inputs)
    }
    routes = tuple(
        _load_route(value, index, project_root)
        for index, value in enumerate(_sequence(raw["routes"], "audio gate routes"), start=1)
    )
    expected_route_ids = {
        "kk-piper-issai-high-v1",
        "kk-mms-kaz-v1",
        "kk-kazemotts-v1",
        "kk-sparktts-v1",
    }
    if len(routes) != 4 or {route.route_id for route in routes} != expected_route_ids:
        raise V4AudioGateError("KK spoof audio gate routes are incomplete or ambiguous.")
    outputs = _mapping(raw["outputs"], "KK spoof audio gate outputs")
    if set(outputs) != {
        "processed_root",
        "runtime_root",
        "output_inventory",
        "output_ready_manifest",
        "output_frozen_manifest",
        "output_receipt",
    }:
        raise V4AudioGateError("KK spoof audio gate output set is invalid.")
    processed_root = _safe_path(outputs["processed_root"], "processed root")
    runtime_root = _safe_path(outputs["runtime_root"], "runtime root")
    output_inventory = _safe_path(outputs["output_inventory"], "output inventory")
    output_ready_manifest = _safe_path(outputs["output_ready_manifest"], "ready manifest")
    output_frozen_manifest = _safe_path(outputs["output_frozen_manifest"], "frozen manifest")
    output_receipt = _safe_path(outputs["output_receipt"], "output receipt")
    if processed_root != PROCESSED_ROOT or runtime_root != RUNTIME_ROOT:
        raise V4AudioGateError("KK spoof audio gate output namespaces changed.")
    selection = _mapping(raw["selection"], "KK spoof audio gate selection")
    if selection != {
        "attempted_per_route": ATTEMPTED_PER_ROUTE,
        "target_per_route": 1_500,
        "reserve_per_route": 300,
        "frozen_ready_per_route": TARGET_PER_ROUTE,
        "order": "target_then_reserve_then_selection_rank",
        "detector_feedback": "prohibited",
        "backfill_outside_frozen_reserve": "prohibited",
    }:
        raise V4AudioGateError("KK spoof audio gate selection policy changed.")
    prohibitions = _mapping(raw["prohibitions"], "KK spoof audio gate prohibitions")
    if prohibitions != {
        "network": True,
        "detector_or_logit_feedback": True,
        "resynthesis": True,
        "output_overwrite": True,
        "new_dataset_search": True,
    }:
        raise V4AudioGateError("KK spoof audio gate prohibitions changed.")
    relative = path.resolve(strict=True).relative_to(project_root).as_posix()
    return Plan(
        path=_safe_path(relative, "plan path"),
        sha256=sha256_file(path),
        created_at=created_at,
        inputs=inputs,
        routes=routes,
        processed_root=processed_root,
        runtime_root=runtime_root,
        output_inventory=output_inventory,
        output_ready_manifest=output_ready_manifest,
        output_frozen_manifest=output_frozen_manifest,
        output_receipt=output_receipt,
        target_per_route=TARGET_PER_ROUTE,
    )


def _load_csv(path: Path, fields: Sequence[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(fields):
                raise V4AudioGateError(f"{label} schema is invalid.")
            return list(reader)
    except OSError as error:
        raise V4AudioGateError(f"Cannot read {label}: {path}") from error


def _verify_route_receipt(route: Route, plan: Plan, project_root: Path) -> None:
    receipt = _json_object(project_root / route.synthesis_receipt.path, f"{route.route_id} receipt")
    route_data = _mapping(receipt.get("route"), f"{route.route_id} receipt route")
    bindings = _mapping(receipt.get("bindings"), f"{route.route_id} receipt bindings")
    outputs = _mapping(receipt.get("outputs"), f"{route.route_id} receipt outputs")
    accounting = _mapping(receipt.get("accounting"), f"{route.route_id} receipt accounting")
    if (
        receipt.get("protocol_id") != "xlsr-sls-model-v4-kk-spoof-synthesis-v1"
        or receipt.get("state") != "route_synthesis_complete_qa_pending"
        or route_data.get("route_id") != route.route_id
        or bindings.get("plan")
        != {
            "path": plan.inputs["synthesis_plan"].path,
            "sha256": plan.inputs["synthesis_plan"].sha256,
        }
        or outputs.get("raw_manifest")
        != {"path": route.raw_manifest.path, "sha256": route.raw_manifest.sha256, "rows": 1_800}
        or outputs.get("synthesis_inventory")
        != {
            "path": route.synthesis_inventory.path,
            "sha256": route.synthesis_inventory.sha256,
            "rows": 1_800,
        }
        or accounting.get("attempted") != 1_800
        or accounting.get("succeeded") != 1_800
        or accounting.get("rejected_runtime") != 0
        or accounting.get("target_succeeded") != 1_500
        or accounting.get("reserve_succeeded") != 300
    ):
        raise V4AudioGateError(f"{route.route_id} synthesis receipt is inconsistent.")


def _load_contexts(
    plan: Plan, project_root: Path, data_root: Path
) -> tuple[dict[str, CandidateContext], list[ManifestRow]]:
    contexts: dict[str, CandidateContext] = {}
    raw_rows: list[ManifestRow] = []
    for route in plan.routes:
        _verify_route_receipt(route, plan, project_root)
        rows = load_manifest(project_root / route.raw_manifest.path)
        inventory = _load_csv(
            project_root / route.synthesis_inventory.path,
            SYNTHESIS_INVENTORY_FIELDS,
            f"{route.route_id} synthesis inventory",
        )
        if len(rows) != len(inventory) != ATTEMPTED_PER_ROUTE:
            raise V4AudioGateError(f"{route.route_id} synthesis packet count changed.")
        by_id = {row.sample_id: row for row in rows}
        if len(by_id) != len(rows):
            raise V4AudioGateError(f"{route.route_id} raw manifest sample IDs are not unique.")
        for item in inventory:
            candidate_id = item["candidate_id"]
            raw = by_id.get(candidate_id)
            try:
                rank = int(item["selection_rank"])
            except ValueError as error:
                raise V4AudioGateError(f"{route.route_id} selection rank is invalid.") from error
            if (
                raw is None
                or candidate_id in contexts
                or item["generator_route_id"] != route.route_id
                or item["terminal_state"] != "succeeded"
                or item["error"]
                or item["raw_relative_path"] != raw.relative_path
                or item["raw_audio_sha256"] != raw.sha256
                or item["target_state"] not in {"target", "reserve"}
                or rank <= 0
                or raw.label != "spoof"
                or raw.language != "kk"
            ):
                raise V4AudioGateError(f"{route.route_id} raw manifest and inventory diverged.")
            contexts[candidate_id] = CandidateContext(
                selection_rank=rank,
                target_state=item["target_state"],
                candidate_id=candidate_id,
                pair_id=item["pair_id"],
                route_id=route.route_id,
                generator_family=item["generator_family"],
                model_id=item["model_id"],
                assigned_voice_id=item["assigned_voice_id"],
                actual_voice_id=item["actual_voice_id"],
                raw_relative_path=item["raw_relative_path"],
                raw_audio_sha256=item["raw_audio_sha256"],
            )
            raw_rows.append(raw)
    if len(contexts) != 7_200 or len(raw_rows) != 7_200:
        raise V4AudioGateError("KK spoof synthesis inputs must contain exactly 7,200 unique rows.")
    require_valid_assets(raw_rows, data_root)
    return contexts, raw_rows


def _decoded_relative_path(raw_sha256: str) -> str:
    _require_sha256(raw_sha256, "raw SHA-256")
    return f"{PROCESSED_ROOT}/{raw_sha256[:2]}/{raw_sha256}.wav"


def _build_tasks(
    contexts: Mapping[str, CandidateContext], data_root: Path
) -> dict[str, V4DecodeTask]:
    tasks: dict[str, V4DecodeTask] = {}
    paths: set[str] = set()
    for context in contexts.values():
        relative = _decoded_relative_path(context.raw_audio_sha256)
        if relative in paths:
            raise V4AudioGateError("KK spoof decode destination collision.")
        paths.add(relative)
        tasks[context.candidate_id] = V4DecodeTask(
            sample_id=context.candidate_id,
            raw_relative_path=context.raw_relative_path,
            raw_sha256=context.raw_audio_sha256,
            source_path=str(resolve_asset_path(data_root, context.raw_relative_path)),
            decoded_relative_path=relative,
            destination_path=str(resolve_asset_path(data_root, relative)),
        )
    if len(tasks) != len(contexts):
        raise V4AudioGateError("KK spoof decode tasks are ambiguous.")
    return tasks


def _add_exact(exact: dict[str, list[str]], digest: str, identity: str) -> None:
    _require_sha256(digest, "historical SHA-256")
    exact.setdefault(digest, []).append(identity)


def _load_history(
    plan: Plan, project_root: Path
) -> tuple[dict[str, tuple[str, ...]], tuple[V4AudioSignature, ...]]:
    source_receipt = _json_object(
        project_root / plan.inputs["source_decode_receipt"].path, "source decode receipt"
    )
    source_outputs = _mapping(source_receipt.get("outputs"), "source decode receipt outputs")
    if (
        source_receipt.get("protocol_id") != "xlsr-sls-model-v4-source-decode-qa-v1"
        or source_receipt.get("state") != "source_train_frozen_15000_kk_spoof_synthesis_authorized"
        or source_outputs.get("decode_inventory")
        != {
            "path": plan.inputs["source_decode_inventory"].path,
            "sha256": plan.inputs["source_decode_inventory"].sha256,
            "rows": 21_598,
        }
        or source_outputs.get("historical_fingerprint_inventory")
        != {
            "path": plan.inputs["historical_fingerprint_inventory"].path,
            "sha256": plan.inputs["historical_fingerprint_inventory"].sha256,
            "rows": 28_400,
        }
    ):
        raise V4AudioGateError("Source decode receipt does not bind audio-gate inputs.")
    source_rows = _load_csv(
        project_root / plan.inputs["source_decode_inventory"].path,
        SOURCE_DECODE_INVENTORY_FIELDS,
        "source decode inventory",
    )
    history_rows = _load_csv(
        project_root / plan.inputs["historical_fingerprint_inventory"].path,
        HISTORY_INVENTORY_FIELDS,
        "historical fingerprint inventory",
    )
    if len(source_rows) != 21_598 or len(history_rows) != 28_400:
        raise V4AudioGateError("Frozen source or historical inventory count changed.")
    exact: dict[str, list[str]] = {}
    signatures: list[V4AudioSignature] = []
    for row in source_rows:
        identity = f"source-v4:{row['candidate_id']}"
        _add_exact(exact, row["raw_audio_sha256"], identity)
        if row["decoded_audio_sha256"]:
            _add_exact(exact, row["decoded_audio_sha256"], identity)
        if row["preparation_status"] == "ready" and row["eligibility_status"] == "eligible":
            signatures.append(
                V4AudioSignature(
                    identity=identity,
                    audio_sha256=row["decoded_audio_sha256"],
                    fingerprint=row["audio_fingerprint_v1"],
                    speech_seconds=float(row["speech_seconds"]),
                )
            )
    for row in history_rows:
        references_value: object = json.loads(row["references"])
        if not isinstance(references_value, list) or len(references_value) != int(
            row["reference_count"]
        ):
            raise V4AudioGateError("Historical fingerprint references are invalid.")
        references = [str(item) for item in references_value]
        for reference in references:
            _add_exact(exact, row["manifest_audio_sha256"], reference)
            if row["fingerprint_status"] == "fingerprinted":
                _add_exact(exact, row["canonical_audio_sha256"], reference)
        if row["fingerprint_status"] == "fingerprinted":
            signatures.append(
                V4AudioSignature(
                    identity=f"historical:{row['manifest_audio_sha256']}",
                    audio_sha256=row["canonical_audio_sha256"],
                    fingerprint=row["audio_fingerprint_v1"],
                    speech_seconds=float(row["speech_seconds"]),
                )
            )
    signature_by_identity = {signature.identity: signature for signature in signatures}
    if len(signature_by_identity) != len(signatures):
        raise V4AudioGateError("Historical audio signatures are ambiguous.")
    return {digest: tuple(sorted(items)) for digest, items in exact.items()}, tuple(signatures)


def _run_decode_workers(
    tasks: Mapping[str, V4DecodeTask],
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
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "stage": "kk_spoof_decode_qa",
                            "completed": len(results),
                            "total": len(tasks),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )


def _decisions(
    contexts: Mapping[str, CandidateContext],
    results: Mapping[str, V4DecodeResult],
    exact: Mapping[str, Sequence[str]],
    signatures: Sequence[V4AudioSignature],
) -> tuple[V4DecodedDecision, ...]:
    if set(results) != set(contexts):
        raise V4AudioGateError("KK spoof decode journal is incomplete.")
    candidates = tuple(
        V4DecodedCandidate(
            selection_rank=context.selection_rank,
            language="kk",
            label="spoof",
            result=results[candidate_id],
        )
        for candidate_id, context in contexts.items()
    )
    return decide_v4_decoded_audio_eligibility(candidates, exact, signatures)


def select_frozen_ids(
    decisions: Sequence[V4DecodedDecision],
    contexts: Mapping[str, CandidateContext],
    target_per_route: int,
) -> tuple[str, ...]:
    """Select only frozen target then reserve order, with no detector-dependent backfill."""

    selected: list[str] = []
    for route_id in sorted({context.route_id for context in contexts.values()}):
        eligible = [
            item
            for item in decisions
            if item.eligibility_status == "eligible"
            and contexts[item.candidate.result.sample_id].route_id == route_id
        ]
        eligible.sort(
            key=lambda item: (
                contexts[item.candidate.result.sample_id].target_state != "target",
                contexts[item.candidate.result.sample_id].selection_rank,
                item.candidate.result.sample_id,
            )
        )
        if len(eligible) < target_per_route:
            raise V4AudioGateError(
                f"KK spoof route {route_id} has {len(eligible)} eligible rows; "
                f"needs {target_per_route}."
            )
        selected.extend(item.candidate.result.sample_id for item in eligible[:target_per_route])
    if len(selected) != len(set(selected)) or len(selected) != 4 * target_per_route:
        raise V4AudioGateError("KK spoof frozen selection is not exactly balanced.")
    return tuple(selected)


def _match_json(matches: Iterable[V4NearAudioMatch]) -> str:
    return json.dumps([asdict(item) for item in matches], sort_keys=True, separators=(",", ":"))


def _inventory_row(
    decision: V4DecodedDecision,
    context: CandidateContext,
    frozen_ids: set[str],
) -> dict[str, object]:
    result = decision.candidate.result
    return {
        "selection_rank": context.selection_rank,
        "target_state": context.target_state,
        "candidate_id": context.candidate_id,
        "pair_id": context.pair_id,
        "generator_route_id": context.route_id,
        "generator_family": context.generator_family,
        "model_id": context.model_id,
        "assigned_voice_id": context.assigned_voice_id,
        "actual_voice_id": context.actual_voice_id,
        "raw_relative_path": result.raw_relative_path,
        "raw_audio_sha256": result.raw_sha256,
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
        "exact_duplicate_of_candidate_id": decision.exact_duplicate_of_candidate_id,
        "historical_exact_match_count": len(decision.historical_exact_matches),
        "historical_exact_matches": json.dumps(
            decision.historical_exact_matches, sort_keys=True, separators=(",", ":")
        ),
        "historical_near_match_count": len(decision.historical_near_matches),
        "historical_near_matches": _match_json(decision.historical_near_matches),
        "within_pool_near_match_count": len(decision.within_pool_near_matches),
        "within_pool_near_matches": _match_json(decision.within_pool_near_matches),
        "frozen_kk_spoof_train_state": "selected"
        if context.candidate_id in frozen_ids
        else "not_selected",
    }


def _publish(
    *,
    plan: Plan,
    project_root: Path,
    data_root: Path,
    raw_rows: Sequence[ManifestRow],
    contexts: Mapping[str, CandidateContext],
    decisions: Sequence[V4DecodedDecision],
    journal: Path,
    created_at: str,
) -> None:
    outputs = tuple(
        project_root / path
        for path in (
            plan.output_inventory,
            plan.output_ready_manifest,
            plan.output_frozen_manifest,
            plan.output_receipt,
        )
    )
    if any(path.exists() or not path.parent.is_dir() for path in outputs):
        raise V4AudioGateError("KK spoof audio gate outputs already exist or have no parent.")
    frozen_ids = set(select_frozen_ids(decisions, contexts, plan.target_per_route))
    raw_by_id = {row.sample_id: row for row in raw_rows}
    ready_rows = tuple(
        replace(
            raw_by_id[item.candidate.result.sample_id],
            relative_path=item.candidate.result.decoded_relative_path,
            sha256=item.candidate.result.decoded_audio_sha256,
            duration_s=item.candidate.result.duration_s,
            original_sr=16_000,
            codec="wav",
            created_at=created_at,
        )
        for item in decisions
        if item.eligibility_status == "eligible"
    )
    frozen_rows = tuple(row for row in ready_rows if row.sample_id in frozen_ids)
    validate_manifest(ready_rows)
    validate_manifest(frozen_rows)
    ledger = load_license_ledger(project_root / plan.inputs["license_ledger"].path)
    validate_manifest_licenses(ready_rows, ledger)
    validate_manifest_licenses(frozen_rows, ledger)
    require_valid_assets(ready_rows, data_root)
    if Counter(contexts[row.sample_id].route_id for row in frozen_rows) != Counter(
        {route.route_id: plan.target_per_route for route in plan.routes}
    ):
        raise V4AudioGateError("KK spoof frozen manifest route balance changed.")
    decision_by_id = {item.candidate.result.sample_id: item for item in decisions}
    counts = Counter(
        (contexts[candidate_id].route_id, decision.eligibility_status, decision.rejection_reason)
        for candidate_id, decision in decision_by_id.items()
    )
    published: list[tuple[Path, Path]] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-kk-spoof-gate-", dir=project_root
        ) as stage_name:
            stage = Path(stage_name)
            staged_inventory = stage / outputs[0].name
            staged_ready = stage / outputs[1].name
            staged_frozen = stage / outputs[2].name
            staged_receipt = stage / outputs[3].name
            with staged_inventory.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=OUTPUT_INVENTORY_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for candidate_id in sorted(
                    contexts,
                    key=lambda item: (
                        contexts[item].route_id,
                        contexts[item].target_state != "target",
                        contexts[item].selection_rank,
                        item,
                    ),
                ):
                    writer.writerow(
                        _inventory_row(
                            decision_by_id[candidate_id], contexts[candidate_id], frozen_ids
                        )
                    )
            write_manifest(staged_ready, ready_rows)
            write_manifest(staged_frozen, frozen_rows)
            receipt = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "created_at": created_at,
                "state": "kk_spoof_train_frozen_5000_training_authorized",
                "bindings": {
                    "plan": {"path": plan.path, "sha256": plan.sha256},
                    **{name: asdict(binding) for name, binding in sorted(plan.inputs.items())},
                    "decode_journal": {
                        "path": journal.relative_to(project_root).as_posix(),
                        "sha256": sha256_file(journal),
                        "rows": len(decisions),
                    },
                    "route_receipts": {
                        route.route_id: asdict(route.synthesis_receipt) for route in plan.routes
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
                        "path": plan.output_inventory,
                        "sha256": sha256_file(staged_inventory),
                        "rows": len(decisions),
                    },
                    "ready_manifest": {
                        "path": plan.output_ready_manifest,
                        "sha256": sha256_file(staged_ready),
                        "rows": len(ready_rows),
                    },
                    "frozen_kk_spoof_train_manifest": {
                        "path": plan.output_frozen_manifest,
                        "sha256": sha256_file(staged_frozen),
                        "rows": len(frozen_rows),
                    },
                },
                "accounting": {
                    "attempted": len(decisions),
                    "decision_counts": {
                        "|".join(key): value for key, value in sorted(counts.items())
                    },
                    "ready_eligible_rows": len(ready_rows),
                    "frozen_route_counts": dict(
                        sorted(
                            Counter(contexts[row.sample_id].route_id for row in frozen_rows).items()
                        )
                    ),
                    "historical_exact_decoded_collisions": sum(
                        bool(item.historical_exact_matches) for item in decisions
                    ),
                    "historical_near_review_candidates_excluded": sum(
                        bool(item.historical_near_matches) for item in decisions
                    ),
                    "within_pool_exact_decoded_duplicates": sum(
                        bool(item.exact_duplicate_of_candidate_id) for item in decisions
                    ),
                    "within_pool_near_review_candidates_excluded": sum(
                        bool(item.within_pool_near_matches) for item in decisions
                    ),
                },
                "selection": {
                    "frozen_ready_per_route": plan.target_per_route,
                    "order": "target_then_reserve_then_selection_rank",
                    "detector_feedback": False,
                    "backfill_outside_frozen_reserve": False,
                },
                "claims": {
                    "technical_decode_qa_vad_performed": True,
                    "decoded_exact_audio_screen_performed": True,
                    "near_audio_screen_performed": True,
                    "training_authorized": True,
                    "speaker_independence": "not_verified_speaker_independent",
                    "new_dataset_search_performed": False,
                },
                "next_gate": (
                    "freeze the combined 20,000-row train manifest and create a separate "
                    "training contract"
                ),
            }
            staged_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for staged, output in zip(
                (staged_inventory, staged_ready, staged_frozen, staged_receipt),
                outputs,
                strict=True,
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
        raise V4AudioGateError(f"Cannot publish KK spoof audio gate packet: {error}") from error


def _ensure_runtime_binding(path: Path, binding: Mapping[str, object]) -> None:
    serialized = json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise V4AudioGateError("Existing KK spoof audio gate runtime binding differs.")
        return
    path.write_text(serialized, encoding="utf-8")


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise V4AudioGateError(f"KK spoof audio gate is already running: {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/research/v4/xlsr_sls_model_v4_kk_spoof_audio_gate_v1.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--created-at", default=_now())
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    arguments = parser.parse_args()
    try:
        if arguments.workers not in range(1, 65):
            raise V4AudioGateError("workers must be between 1 and 64.")
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        plan_path = arguments.plan.resolve(strict=True)
        plan_path.relative_to(project_root)
        plan = load_plan(plan_path, project_root)
        contexts, raw_rows = _load_contexts(plan, project_root, data_root)
        tasks = _build_tasks(contexts, data_root)
        exact, signatures = _load_history(plan, project_root)
        runtime = project_root / plan.runtime_root
        runtime.mkdir(parents=True, exist_ok=True)
        binding = {
            "schema_version": 1,
            "protocol_id": f"{PROTOCOL_ID}-runtime",
            "plan": {"path": plan.path, "sha256": plan.sha256},
            "inputs": {name: asdict(value) for name, value in sorted(plan.inputs.items())},
            "routes": {route.route_id: asdict(route.synthesis_receipt) for route in plan.routes},
            "processed_root": plan.processed_root,
            "workers": arguments.workers,
        }
        _ensure_runtime_binding(runtime / "binding.json", binding)
        lock_path = runtime / "gate.lock"
        lock = _acquire_lock(lock_path)
        try:
            journal = runtime / "kk_spoof_decode_qa.jsonl"
            results = load_v4_decode_journal(journal, tasks)
            _run_decode_workers(tasks, results, journal, arguments.workers)
            decisions = _decisions(contexts, results, exact, signatures)
            _publish(
                plan=plan,
                project_root=project_root,
                data_root=data_root,
                raw_rows=raw_rows,
                contexts=contexts,
                decisions=decisions,
                journal=journal,
                created_at=arguments.created_at,
            )
        finally:
            os.close(lock)
            lock_path.unlink(missing_ok=True)
    except (
        LicenseLedgerError,
        ManifestError,
        OSError,
        ValueError,
        V4AudioGateError,
    ) as error:
        issues = list(error.issues) if isinstance(error, LicenseLedgerError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "receipt": plan.output_receipt,
                "frozen_rows": 4 * plan.target_per_route,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
